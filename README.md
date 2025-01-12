# ShowHowTo: Generating Scene-Conditioned Step-by-Step Visual Instructions

### [[Project Website :dart:]](https://soczech.github.io/showhowto/)&nbsp;&nbsp;&nbsp;[[Paper :page_with_curl:]](https://arxiv.org/abs/2412.01987)&nbsp;&nbsp;&nbsp;[Code :octocat:]

This repository contrains code for the paper [ShowHowTo: Generating Scene-Conditioned Step-by-Step Visual Instructions](https://arxiv.org/abs/2412.01987).


## Run the model on your images and prompts
1. **Environment setup**
   - Use provided `Dockerfile` to build the environment or install the [packages](https://github.com/soCzech/ShowHowTo/blob/main/Dockerfile) manually.
     ```
     docker build -t showhowto .
     docker run -it --rm -v $(pwd):$(pwd) -w $(pwd) --gpus=1 showhowto:latest bash
     ```
   - The code, as written, requires a GPU.

2. **Download ShowHowTo model weights**
   - Use `download_weights.sh` script or download the [ShowHowTo weights](https://data.ciirc.cvut.cz/public/projects/2024ShowHowTo/weights/) manually.

3. **Get predictions**
   - Run the following command to get example predictions.
     ```
     python predict.py --ckpt_path ./weights/showhowto_2to8steps.pt 
                       --prompt_file ./test_data/prompt_file.txt
                       --unconditional_guidance_scale 7.5
     ```
   - To run the model on your images and prompts, replace `./test_data/prompt_file.txt` with your prompt file.


## Training
1. **Environment setup**
   - Use the same environment as for the prediction (see above).

2. **Download DynamiCrafter model weights**
   - Use `download_weights.sh` script or download the [DynamiCrafter weights](https://huggingface.co/Doubiiu/DynamiCrafter/blob/main/model.ckpt) manually.

3. **Get the dataset**
   - To replicate our experiments on the ShowHowTo dataset, see below, or use your own dataset.
   - The dataset must have the following directory structure.
     ```
     dataset_root
     ├── prompts.json
     └── imgseqs
         ├── <sequenceid>.jpg
         │   ...
         └── ...
     ```
     There can be multiple directories with names starting with `imgseqs`. 
   - The `promts.json` file must have the following structure.
     ```
     {
       "<sequenceid>": ["prompt for the 1st frame", "prompt for the 2nd frame", ...],
       ...
     }
     ```
   - The sequence image `<sequenceid>.jpg` must be of width `N*W` (`W` is width of each image in the sequence) and arbitrary height `H`.
     The number of images in the sequence `N` must match the length of the prompt list in the `prompts.json` file.
4. **Train**
   - Run the training code.
     ```
     python train.py --local_batch_size 2
                     --dataset_root /path/to/ShowHowToTrain
                     --ckpt_path weights/dynamicrafter_256_v1.ckpt
     ```
   - We trained on a single node with 8 GPUs with the batch size of 2 videos per GPU. Be advised, that more than 40 GB of VRAM per GPU may be required to train with batch size larger than 1.


## Dataset
You can download the ShowHowTo dataset using the `download_dataset.sh` script. To also download the image sequences from our servers, you need username and password.
You can obtain it by sending an email to *tomas.soucek at cvut dot cz* specifying your name and affiliation. Please use your institutional email (i.e., not gmail, etc.).

You can also extract the dataset from the raw original videos with the following steps.

1. **Download the HowTo100M videos and the ShowHowTo prompts**
   - The list of all video ids for both the train set and test set can be found [here](https://data.ciirc.cvut.cz/public/projects/2024ShowHowTo/dataset/).
   - For each video, the `keyframes.json` file contains information on which video frames are part of the dataset.
   - You can find there also the prompts for each video in `prompts.json` file.
2. **Extract the video frames of the ShowHowTo dataset**
   - To extract the frames from the videos, we used ffmpeg v7.0.1 with the following function.
     ```python
     def extract_frame(video, start_sec, frame_idx, width, height):
         ffmpeg_args = ['ffmpeg', '-i', video, '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                        '-vf', f'fps=5,select=gte(t\\,{start_sec}),select=eq(n\\,{frame_idx})',
                        '-s', f'{width}x{height}', '-vframes', '1', 'pipe:']
         video_stream = subprocess.Popen(ffmpeg_args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
         
         in_bytes = video_stream.stdout.read(width * height * 3)
         return np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
     ```
     The function arguments are: `video` is the path to the video, `start_sec` and `frame_idx` are the values from the `keyframes.json` and `width` and `height` specify the output image size (we used the native video resolution here).
3. **Prepare the image sequences**
   - Concatenate all frames from a video in the horizontal dimension and place the resulting concatenated image into `dataset_root/imgseqs/<sequenceid>.jpg`. The `<sequenceid>` is the YouTube video id.




## Citation
```bibtex
@article{soucek2024showhowto,
    title={ShowHowTo: Generating Scene-Conditioned Step-by-Step Visual Instructions},
    author={Sou\v{c}ek, Tom\'{a}\v{s} and Gatti, Prajwal and Wray, Michael and Laptev, Ivan and Damen, Dima and Sivic, Josef},
    month = {December},
    year = {2024}
}
```
