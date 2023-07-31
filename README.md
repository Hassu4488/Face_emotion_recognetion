Facial Expression Recognition

Personal Detail:
Name : Hasnain ullah
Section :03
Email : 4488hassu@gmail.com
GithHub profile : https://github.com/Hassu4488

Introduction:

There are some things which were very easy for humans; on the other hand, it is quite difficult for a machine. Just like to judge the emotions of a person are very easy for humans but, the machine needs human efforts to process this task. So for this purpose I have completed a project in last week, in which I made a model which is trained to judge the emotions from the faces of a person.
This project is completed by using the technology “computer vision”, which is the branch of Artificial intelligence (AI). Computer vision involves in the development of algorithms and models which learns the useful information from the images and videos. Deep learning as the subfield of Artificial intelligence plays a vital role in computer vision tasks.
So far, there are many programming languages of coding like., Java script but I choose python as programming language for my project because of its built in libraries which provided me the easy path to complete my facial expression recognition model.


Project Overview:

Emotion detection refers to the identification of emotions from the faces and voices of humans. In the past, emotions were detected by the hand craft features and the rule-base approaches in which features were extracted from the input data and use predefined rules to identify emotions. While in the era of deep learning, by using neural networks features are extracted automatically from the raw data. 
Scope:
The facial emotion recognition has wild range scope. The scope includes: emotion-aware-human-computer, sentiment analysis in social media, and emotional feedback in virtual reality environments.
Duration:
Facial emotion recognition task is very complex task because through the world there are different peoples with different casts and having different faces and they express their emotions in different ways that’s why we have to train our model in large dataset which takes much time and to train the model with well accuracy also time consuming task.
In my case it took almost 15 hours (in 4GB ram with i5 processor)  
	Through out project we emphasize the importance of interpretability and practicality. While deep learning models can be complex, we aim to provide insights into the model predictions and explore ways to enhance model interpretability. 
Project Summary:
Our project involves the following steps:
•	Data collection and preprocessing:
I used the pre built dataset from google after that I pushed it for pre processing.
https://drive.google.com/drive/folders/1SDcI273EPKzzZCPSfYQs4alqjL01Kybq 
•	Model Architecture:
I design a deep learning model, based on convolutional neural network to learn and extract meaningful features from the facial images.
I also used a pre trained model (VGG 16) which helped to improve generalization.

•	Hyperparameters Tuning:
The fine-tuning was used in model to optimized performance. The data augmentation and dropout are also used generalize data and to reduce the overfitting.

•	Evaluation and testing :
In the training phase the data was splitted into training and test data. In the evaluation and testing phase the data is evaluated in different test set to assess its accuracy and effectiveness in recognizing face expression.

•	Results and Evaluations:
I have been treated this model in less computational power. In my machine it gives 70% accuracy.

•	Analysis of Results:
The confusion matrix showed that the model performed well in recognizing Happy and Neutral expression but struggled with Disgust and Fear emotions.

•	Suggestions for further improvement:

•	Collect more divers data.
•	Use data augmentation in large scale
•	Test different pre trained models
•	Test  on different learning rate
      
Literature review:
There are two famous articles about face recognition one is written by lead author is Dr. A. K. Sangaiah and his team:
https://www.mdpi.com/1424-8220/22/16/6105 
Work: The authors propose a novel approach to facial expression recognition using a deep learning model called a convolutional neural network (CNN). The CNN is trained on a dataset of facial images with annotated expressions. The authors also use a technique called transfer learning, which involves training the CNN on a large dataset of images with other labels, and then fine-tuning the CNN for facial expression recognition.
Data: The authors used the FER2013 dataset for training and testing their model. The FER2013 dataset contains over 3,500 facial images with annotated expressions. The expressions in the FER2013 dataset include happiness, sadness, anger, surprise, fear, and disgust.
Accuracy Reported: The authors report that their model achieved an accuracy of 94.5% on the FER2013 dataset. This is a significant improvement over the accuracy of previous models that have used CNNs for facial expression recognition.
Pros: The main advantage of the proposed approach is that it achieves high accuracy while being computationally efficient. The CNN is able to learn features from facial images that are relevant to facial expression recognition, which allows it to achieve high accuracy.
Cons: One potential drawback of the proposed approach is that it may not be as robust to variations in pose and illumination as other approaches. However, the authors argue that their model is still able to achieve high accuracy under these conditions..
The second article was written by  Dr. Zhi-Hong Zhang and his team:
https://www.nature.com/articles/s41598-023-35446-4
Work:
The authors propose a novel approach to facial expression recognition using a convolutional neural network (CNN) with attention. The CNN is trained on a dataset of facial images with annotated expressions. The attention mechanism is used to focus the CNN on the most important parts of the face when making predictions.
Data:
The authors used the AffectNet dataset for training and testing their model. The AffectNet dataset contains over 400,000 facial images with annotated expressions. The expressions in the AffectNet dataset include happiness, sadness, anger, surprise, fear, disgust, contempt, and neutral.
Accuracy Reported:
The authors report that their model achieved an accuracy of 93.5% on the AffectNet dataset. This is a significant improvement over the accuracy of previous models that have used CNNs for facial expression recognition.
Pros:
The main advantage of the proposed approach is that it achieves high accuracy while being computationally efficient. The attention mechanism helps the CNN to focus on the most important parts of the face, which reduces the amount of computation required.
Cons:
One potential drawback of the proposed approach is that it may not be as robust to variations in pose and illumination as other approaches. However, the authors argue that their model is still able to achieve high accuracy under these conditions.

•	Conclusion:
The project’s next objectives include improving the data using more advanced image processing techniques to improve the model’s performance. Further research into more complex model structures and advancements in methodology may lead to even greater results.


 
 

