# ShowHowTo: Generating Scene-Conditioned Step-by-Step Visual Instructions

### [[Project Website :dart:]](https://soczech.github.io/showhowto/)&nbsp;&nbsp;&nbsp;[[Paper :page_with_curl:]](#TODO)&nbsp;&nbsp;&nbsp;[Code :octocat:]

This repository contrains code for the paper [ShowHowTo: Generating Scene-Conditioned Step-by-Step Visual Instructions](#TODO).


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
_The training code will be released in the coming weeks._


## Dataset
_The dataset will be released in the coming weeks._


## Citation
```bibtex
@article{soucek2024showhowto,
    title={ShowHowTo: Generating Scene-Conditioned Step-by-Step Visual Instructions},
    author={Sou\v{c}ek, Tom\'{a}\v{s} and Gatti, Prajwal and Wray, Michael and Laptev, Ivan and Damen, Dima and Sivic, Josef},
    month = {December},
    year = {2024}
}
```
