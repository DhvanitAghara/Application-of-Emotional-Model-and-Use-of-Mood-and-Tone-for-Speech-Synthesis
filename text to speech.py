# Import the required module for text  
# to speech conversion 
from gtts import gTTS 
  
# This module is imported so that we can  
# play the converted audio 
import os 
  




# The text that you want to convert to audio 
#mytext= "I dont want to talk, i am very sad."

file_name = "G:/Bisag Internship/Generate-Audio-From-Emotions-master/Text-classification/text.txt"
file=open(file_name,'r')
mytext=file.read()




# Language in which you want to convert 
language = 'en'
  
# Passing the text and language to the engine,  
# here we have marked slow=False. Which tells  
# the module that the converted audio should  
# have a high speed 
myobj = gTTS(text=mytext, lang=language, slow=False) 
  
# Saving the converted audio in a mp3 file named 
# welcome  
myobj.save("speech.mp3") 
  
# Playing the converted file 
os.system("speech.mp3") 