# Sentiment Analysis using RNN: 
- Sentiment analysis is typically a classification problem.
- However for a larger data set RNN can provide a better prediction than typical Machine Learning approach.  


## Output: 

- This is training vs validation loss.
- Since we have around 25K reviews, the network is over-fitting beyond 5 epochs       

![Alt text](img/plot1.png?raw=true "Output") 

## RNN Model Architecture:

- We need to calculate the loss against only the last sigmoid unit.
- The embedding layer will be trained automatically.

![Alt text](img/network_diagram.png?raw=true "Architecture")

## Hyperparameters:

- Input Dataset                 : 25K Reviews
- Train/Validation/Test Split   : 80%/10%/10%
- Number of hidden layers       : 256
- LSTM Layer ( stacking )       : 2
- Learning Rate                 : 0.001
- Batch Length                  : 50
- epochs                        : 4

## Loss: 

- Average Train Loss        :   0.6725
- Average Validation Loss   :   1.2836
- Average Test Loss         :   0.529
    - Test Accuracy         :   80% 
 
## Dataset:

-  Extract the reviews.zip in the dataset folder
   
## Related work:

- Here is another work for Sentiment Analysis using Machine Leaning.
- Visualization:
    - https://adeveloperdiary.github.io/PSL_Project4/index.html
    ![Alt text](img/related.png?raw=true "Output")
    - Source : https://github.com/adeveloperdiary/PSL_Project4
- Machine Leaning Model:
    - TBA (To be added to github)
        
