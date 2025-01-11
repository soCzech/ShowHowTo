import os
import tqdm
import torch
import random
import socket
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

from datetime import datetime
from omegaconf import OmegaConf
from einops import rearrange, repeat

from lvdm.utils import instantiate_from_config, load_model_checkpoint, get_cosine_schedule_with_warmup, AverageMeter
from video_dataset import sequence_collate, RepeatedDataset, ShowHowToDataset


def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def main(args):
    ngpus_per_node = torch.cuda.device_count()
    node_count = int(os.environ.get("SLURM_NPROCS", "1"))
    node_rank = int(os.environ.get("SLURM_PROCID", "0"))
    job_id = os.environ.get("SLURM_JOBID", "".join([str(random.randint(0, 9)) for _ in range(5)]))

    dist_url = "file://{}.{}".format(os.path.realpath("distfile"), job_id)
    print(f"Hi from node {socket.gethostname()} ({node_rank}/{node_count} with {ngpus_per_node} GPUs)!", flush=True)

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=({
        "ngpus_per_node": ngpus_per_node,
        "node_count": node_count,
        "node_rank": node_rank,
        "dist_url": dist_url,
        "job_id": job_id
    }, args))


def main_worker(local_rank, cluster_args, args):
    world_size = cluster_args["node_count"] * cluster_args["ngpus_per_node"]
    global_rank = cluster_args["node_rank"] * cluster_args["ngpus_per_node"] + local_rank
    dist.init_process_group(
        backend="nccl",
        init_method=cluster_args["dist_url"],
        world_size=world_size,
        rank=global_rank,
    )

    if global_rank == 0:
        store_dir = "logs/" + datetime.strftime(datetime.now(), "%Y-%m-%d_%H%M%S")
        for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
            print(f"# {k}: {v}")
        print(f"# effective_batch_size: {world_size * args.local_batch_size}", flush=True)

    ###############
    # DATASET
    ###############
    n_epochs = 200
    save_every_n_epochs = 1

    train_ds = []
    for i in range(2, args.max_seq_len + 1):
        train_ds.append(RepeatedDataset(
            ShowHowToDataset(args.dataset_root, video_length=i), epoch_len=16000))

    train_samplers = [None for _ in train_ds]
    if world_size > 1:
        train_samplers = [torch.utils.data.distributed.DistributedSampler(ds, shuffle=True, drop_last=True) for ds in train_ds]

    train_ds_iters = [torch.utils.data.DataLoader(
        ds, batch_size=args.local_batch_size, shuffle=world_size == 1, drop_last=True, num_workers=1,
        pin_memory=True, sampler=train_sampler, collate_fn=sequence_collate) for ds, train_sampler in zip(train_ds, train_samplers)]

    ###############
    # MODEL
    ###############
    learning_rate = 2e-5

    config = OmegaConf.load("./configs/inference_256_v1.1.yaml")["model"]
    model = instantiate_from_config(config)
    model = load_model_checkpoint(model, args.ckpt_path)

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    model_parallel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    print(f"Model distributed to gpu {global_rank}!", flush=True)

    ###############
    # OPTIMIZER
    ###############
    parameters2train = model_parallel.module.model.parameters()

    optim = torch.optim.AdamW(parameters2train, lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optim, len(train_ds_iters[0]), len(train_ds_iters[0]) * n_epochs)
    loss_metric = AverageMeter()

    for epoch in range(1, n_epochs + 1):
        if world_size > 1:
            for train_sampler in train_samplers:
                train_sampler.set_epoch(epoch)

        iterator = tqdm.tqdm(train_ds_iters[-1]) if global_rank == 0 else train_ds_iters[-1]
        other_iterators = [iter(ds) for ds in train_ds_iters[:-1]]
        for video_frames, prompts in iterator:
            # gather data for all lengths
            iterator_data = [(video_frames, prompts)]
            for other_iterator in other_iterators:
                iterator_data.append(next(other_iterator))

            for video_frames, prompts in iterator_data:
                B, C, T, H, W = video_frames.shape
                frame_stride = torch.ones((B,), dtype=torch.long, device=model.device)

                with torch.no_grad():
                    img_emb = model.image_proj_model(model.embedder(video_frames[:, :, 0].to(model.device)))
                    text_emb = model.get_learned_conditioning(prompts)
                    z = get_latent_z(model, video_frames.to(model.device))

                tB, tL, tC = text_emb.shape
                if tB != B:  # in case we have multiple prompts for a single video
                    assert B * T == tB, f"{B} * {T} != {tB}"
                    img_emb = img_emb.repeat_interleave(repeats=T, dim=0)
                cond = {
                    "c_crossattn": [torch.cat([text_emb, img_emb], dim=1)],
                    "c_concat": [repeat(z[:, :, :1], 'b c t h w -> b c (repeat t) h w', repeat=T)]
                }

                t = torch.randint(0, model.num_timesteps, (z.shape[0],), device=model.device).long()
                noise = torch.randn_like(z)
                x_noisy = model.q_sample(x_start=z, t=t, noise=noise)

                model_output = model_parallel(x_noisy, t, cond, fs=frame_stride)

                loss = torch.nn.functional.mse_loss(noise, model_output, reduction='none')
                loss = loss.mean([1, 2, 3, 4])
                loss = loss.mean()

                optim.zero_grad()
                loss.backward()  # DistributedDataParallel does gradient averaging, i.e. loss is x-times smaller when trained on more GPUs
                optim.step()
                loss_metric.update(loss.item())

            scheduler.step()

        if global_rank == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss_metric.value:.4f}", flush=True)

            os.makedirs(store_dir, exist_ok=True)
            with open(os.path.join(store_dir, "losses.txt"), "a") as f:
                f.write(f"{epoch:03d}{loss_metric.value:12.4f}\n")
            loss_metric.reset()

            if epoch % save_every_n_epochs == 0:
                torch.save(model_parallel.module.state_dict(), os.path.join(store_dir, f"model_{epoch:03d}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_batch_size", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="./weights/dynamicrafter_256_v1.ckpt")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=8)

    main(parser.parse_args())
