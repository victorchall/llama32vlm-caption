# Llama 3.2 captioning script with in-context learning

This is a Llama 3.2 VLM-only version of another script I wrote for other VLM models here: https://github.com/victorchall/EveryDream2trainer/blob/main/caption.py

This is extracted to avoid compatibility that arise from recent transformers updates that break Cog but are required for Llama 3.2. This will likely be rolled back into the main script in ED2 repo at some point in the future when I decide to deprecate Cog as Cog is not being maintained anyway. 

Additional info is found here: https://github.com/victorchall/EveryDream2trainer/blob/main/doc/CAPTION_COG.md

This version is simply hard-coded to use Llama 3.2 Vision 11B and use current (as of 2024-10-12) versions of the Huggingface transformers package.

Example usage:

Caption/describe the image using greedy sampling:

    python caption.py --image_dir "/mnt/my_images" --prompt $"Describe this image. Do not exceed 60 words." --starts_with "This image showcases" --remove_starts_with --num_beams 1 

Caption/describe the image and add a hint from each image's .json sidecar file (`--prompt_plugin from_image_json`), and skip the image if it already has a .txt file label (`--no_overwrite`):
    
    python caption.py --image_dir "/mnt/my_images" --prompt $"Describe this image. Do not exceed 60 words." --starts_with "This image showcases" --remove_starts_with --num_beams 1 --prompt_plugin from_image_json --no_overwrite

1 beam with short outputs is about 22GB of VRAM.

See the [caption readme](https://github.com/victorchall/EveryDream2trainer/blob/main/doc/CAPTION_COG.md#prompt-modification-plugins) for more information on `--prompt_plugin` options. There are various plugins ready to go and *these plugins together any metadata you have on your images can substantially improve accuracy*. It should be quite easy to write new ones simply by giving an LLM (ChatGPT, Llama 70B, etc) one of the existing plugin classes and asking it to write a derived one to your specification.

`requirements.txt` is supplied to setup your own venv or conda env, but different operating systems (i.e. windows) may need to install cuda-included versions of Torch and/or have cuda installed.  Windows is untested.

You can try this if you're having troubles:

    pip install torch>=2.4.1+cu121 --extra-index-url "https://download.pytorch.org/whl/cu121"

Windows users can also use WSL with an Nvidia/Pytorch/Cuda image, various cloud GPU services, Google Colab Pro (>=24GB VRAM), etc. 

## Hugginface login

Since Llama models are gated access on Huggingface, you'll first need to go to the [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) site while logged into Hugginface and accept their terms.  You'll then need to make sure your terminal is authenticated to Huggingface for the model to load properly.

On the terminal with the venv activated, try running this first to see if you are logged in:

    huggingface-cli whoami

You should see something like this:

    myusername
    orgs: myorg

If you are not authenticated, run this and paste in an [access token](https://huggingface.co/docs/hub/en/security-tokens), ([link to create](https://huggingface.co/settings/tokens)).

    huggingface-cli login

Your system should remember you from that point forward, though if you're running on a rented instance you'll have to login every time you start a new rented instance.