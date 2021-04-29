# Urban-Street-Cleanliness-Assessment-using-Mobile-Edge-Computing-and-Deep-Learning
During the process of smart city construction, city managers always spend a lot of energy and money for cleaning street garbage due to the random appearances of street garbage. Consequently, visual street cleanliness assessment is particularly important. However, the existing assessment approaches have some clear disadvantages, such as the collection of street garbage information is not automated and street cleanliness information is not real-time. To address these disadvantages, this paper proposes a novel urban street cleanliness assessment approach using mobile edge computing and deep learning. First, the high-resolution cameras installed on vehicles collect the street images. Mobile edge servers are used to store and extract street image information temporarily. Second, these processed street data is transmitted to the cloud data center for analysis through city networks. At the same time, Faster Region-Convolutional Neural Network (Faster R-CNN) is used to identify the street garbage categories and count the number of garbage. Finally, the results are incorporated into the street cleanliness calculation framework to ultimately visualize the street cleanliness levels, which provides convenience for city managers to arrange clean-up personnel effectively.

Steps to Run the Project on Anaconda:

commands for installing necessary packages:

conda create -n tf python=3.6
y
activate tf
conda install keras
y
pip install opencv_contrib_python
pip install sklearn
pip install pandas
pip install seaborn
pip install pil
pip install pillow
pip install scikit-image
pip install imutils
pip install tqdm

Steps to execute the Project:

activate tf
folder name has to be given followed by ":" -- example--  e:
Path of the Main file has to be given excluding Main -- example-- \Project code\Garbage Detection
python Main.py

After this a user interface opens where the user needs to provide the path of the file containing street images i.e., test_images file. 
Next step is to click on the Predict button.
The Garbage Detection of the available images is performed and the necessary message is sent to the City Administrator for the future use.
