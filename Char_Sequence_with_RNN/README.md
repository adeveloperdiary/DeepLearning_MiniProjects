# Generate Python Source Code using RNN: 
- Developed a simple character sequence model using RNN and LSTM. 
- Used scikit-learn open source code as input data to generate python source code.

## Output: 

- Here is one sample of generated output:  

![Alt text](output/screenprint.png?raw=true "Title")

## Hyperparameters:

- Input file length         : ~9M words
- Train/Validation Split    : 90%/10%
- Number of hidden layers   : 512
- LSTM Layer ( stacking )   : 2
- Learning Rate             : 0.001
- Batch Length              : 200
- Sequence Length           : 128
- epochs                    : 50

## Loss: 

- Average Train Loss        :   0.6725
- Average Validation Loss   :   1.2836
 
## Dataset:

-  scikit-learn github repo.
    - https://github.com/scikit-learn/scikit-learn
