# Conditional DCGAN (CDCGAN)
The `GAN/CDCGAN` folder contains the Conditional Deep Convolutional GAN (CDCGAN) implementation. All file names within this section will refer to files  within this folder unless otherwise specified.

The GAN was implemented using PyTorch and the model files can be found in `model` subdirectory. We have also saved our best trained models for generating Mario game levels in the `trained_model` directory.

Code adapted from <a href = https://github.com/CS527Applied-Machine-Learning-for-Games/GAN-Based-Game-Level-Generation/blob/master/docs/Technical%20Report.pdf > Banerjee, S., et al. (2021). "GAN-Based-Game-Level-Generation."</a>


## Data

The video game level data is stored in `levels.json` file. It contains 14 x 28 tile unit frames of Super Mario Bros 2 levels.

To train our CDCGAN, we split every frame in half -- the first half is used as a condition and the second half is generated. This is done in the `MarioDataset` class in `data_loader.py`.

## Instructions for training your own GAN

To train the GAN you can run the following command in the `GAN/CDCGAN` directory:

```
python train.py
```

If you want to use cuda then append the `--cuda` argument to the above. Hyperparameters can be changed in `train.py`.  A number of filenames in the code may need to be set mmanually depending on what name you would like the files to be saved as. By default, they will save in a directory called `samples`.


## Generating levels using trained model
Make sure you've installed our `image_gen` package included in the repository by running 
```
pip install -e .
```
while in `GAN/ImageGenerators/image_gen` directory.

To generate levels using a trained model you can run
```
python get_level.py
```
Within `get_level.py`, you need to include the path to your trained model. By default, the parameter for this is set to a placeholder:
```
netG.load_state_dict(torch.load("YOUR_MODEL_PATH"))
```
A number of filenames in the code may need to be set mmanually depending on what name you would like the files to be saved as. By default, they will save in a directory called `samples`.

# Latent Space Exploration (LSE) using CMA-ES

We perform Latent Space Exploration on the noise input to our Conditional DCGAN. The idea is to learn a mapping between certain areas of this noise to certain features. 

Our implementation of Covariance matrix adaptation evolution strategy (CMA-ES) can be found in `cmaes.py`. 

## Fitness functions
CMA-ES utilizes a fitness function to judge how good a certain member of the sample set ("population") is. 

`lse.py` contains a hand-crafted fitness function `matrix_eval` that gathers numerous tile frequencies in certain regions of the level, and uses these metrics to optimize for different structural features in the level, e.g. increasing the `sky-tile` count. 

## Instructions to perform LSE

To perform LSE on a trained CDCGAN run
```
python lse.py
```
This defaults to `1000` population members per iteration, you can configure that by changing the `population_size` parameter in `__main__`.
You can also configure the level samples that are generated per member by setting `samples_per_member` parameter. 

To define a custom fitness function, you can define a new function which takes in the generated level as a `numpy array` and returns a custom fitness measure for the level.

## Instructions for level generation with feature-noise parameters
After performing LSE to find the best parameters for certain features, we can then orchestrate the level generation by running `demo_lse.py`, which loads the saved noise parameters with the trained generator and then presents the user with a menu to choose the features they want in next `frame_count=6` frames of the level.

Command to run:

```
python demo_lse.py
```

# NES emulator 
## Instructions to convert generated levels to npy 
Using a trained GAN, we can generate levels by running
```
python get_level.py
```
specifically the `save_full_level_np` method. Levels will be saved as a `.npy` file. 
Note that  `save_full_level_np` will save two versions of the level, one of which will have a 13 at the end of the name. Use the one with 13 at the end. The file must then be renamed `new_level.npy` for conversion to work. 

## Instructions to use emulator
To install the NES emulator system, you can clone the `custom_level` branch of the following repository by running 
```
git clone https://github.com/CameronChurchwell/nes-py.git
```

Install the emulator package by running
```
pip install -e .
```
To run the generated level as a Gym environment, run

```
python example.py
```

Code from <a href = https://github.com/CameronChurchwell/nes-py.git > Kauten, C. and Churchwell, C.</a>

# Reinforcement learning
The RL agent can be trained by running `Mario_RL.ipynb` (e.g. in Google Colab), setting the desired episode number in the `run()` function. Specifically, the cell containing
```
run(training_mode = True, pretrained = False)
```
will train the model.

A pretrained model can be tested by running the cell containing
```
run(training_mode = False, pretrained = True)
```

Code adapted from <a href = https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/ > Grebenisan, A. (2021). “Building a Deep Q-Network to Play Super Mario Bros."</a>


