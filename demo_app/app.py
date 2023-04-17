import pickle
from prophet import Prophet
import gradio as gr
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# File Paths
model_paths = [ 'tf_Junction_0.sav', 'tf_Junction_1.sav',
               'tf_Junction_2.sav', 'tf_Junction_3.sav' ]

# Loading the files
models = {i:pickle.load(open(model_path, 'rb')) for i, model_path in zip(range(1,5), model_paths)}


# Util function

def forecast(J_class, num_days):
  
  # extracting the model
  model = models[int(J_class)]

  # Creating next days to forecast
  future = model.make_future_dataframe(periods=num_days)
  
  # forcasting
  predicts = model.predict(future).iloc[-num_days:,:]
  
  return predicts

def clean_buffer():
  for img in glob.glob('*.png'):
    os.remove(img)

def visualizate(predicts):

  fig = plt.figure()
  sns.lineplot(data = predicts, x="ds", y="yhat", label="yhat", color='red')
  sns.lineplot(data = predicts, x="ds", y="yhat_lower", label="yhat_lower", color='green')
  sns.lineplot(data = predicts, x="ds", y="yhat_upper", label="yhat_upper", color='orange')

  # Save the figure to a buffer
  clean_buffer()

  time = datetime.now().strftime("%H:%M:%S")
  save_path = f"temp_fig_{time}.png"
  
  fig.savefig(save_path, format='png')

  return save_path

def predict(J_class, num_days):
  return visualizate(forecast(J_class, num_days))

# Creating the gui component according to component.json file
Junction_no = gr.inputs.Dropdown(["1","2","3","4"], default="1", label="Junction No:")
num_day =gr.inputs.Slider(minimum=1, maximum=30, default=5, step=1, label="Number of next days to forcest")
inputs = [Junction_no, num_day]

demo_app = gr.Interface(predict, inputs, "image" ).launch()


# Launching the demo
if __name__ == "__main__":
    demo_app.launch()
