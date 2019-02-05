import datetime
from PIL import Image
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import plotly.graph_objs as go
import dash_reusable_components as drc
import base64
import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_css = [
    # Normalize the CSS
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    # Fonts
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto"
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    # For production
    "https://cdn.rawgit.com/xhlulu/0acba79000a3fd1e6f552ed82edb8a64/raw/dash_template.css",
    # Custom CSS
    "https://cdn.rawgit.com/xhlulu/dash-image-processing/1d2ec55e/custom_styles.css",
]

app = dash.Dash(__name__, external_stylesheets=external_css)

colors = {

    'background': '#FFFFFF',
    'text': "blue"#'#7FDBFF'
}


styles1 = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

theme = {
    'dark': False,
    'detail': '#007439',
    'primary': '#00EA64', 
    'secondary': '#6E6E6E'
}


app.scripts.config.serve_locally = True

app.layout = html.Div(style={'backgroundColor': "#000403","height":"600%"}, children=[
    html.H1(
        children='Image Captioning',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Caption: Generates a Image Description', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
	html.Div(
		html.Div(
			children=[
				html.Div(
					className="two columns",
							children=[
								html.Div(children=[
    										dcc.Upload(
        										id='upload-image',
        										children=html.Div([
            										'Drag and Drop or  ',
            										html.Button('Select Files',style = {
													"color":"Green"
															})
        										]),
        										style={
           										 'width': '100%',"color":"Blue",

            										'height': '60px',

            										'lineHeight': '60px',

            										'borderWidth': '1px',

            										'borderStyle': 'dashed',

            										'borderRadius': '5px',

           										 'textAlign': 'center',

           										 'margin': '10px'
       											 },
        										# Allow multiple files to be uploaded
      											  multiple=True
   								 ),
    								html.Div(id='output-image-upload'),

								])
							])
						])
					)
			])


def parse_contents(contents, filename, date):
    res =  main('/root/ImageCaptioning/data/resized2017/'+filename)
    #print(res)
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents,style = {'position': 'inline-block',
	
					"width":"25%","height":"30%", 
	
					"margin-left":"500px", 

					"margin-bottom":"0px", 

					"padding":"0px",

					'text-align':"center",

					'horizontal-align': 'middle',

					'position' : 'relative'
					}),
        
        html.Hr(),
	
        html.Div('Raw Content:',style = {'color':"Blue",
					'textAlign': 'center'}),
#        html.P( res+ '...', style={
#            'whiteSpace': 'pre-wrap',
#            'wordBreak': 'break-all', 'margin-left':"380px",'color':"Black"
#        }),
		dcc.Textarea(
    placeholder='Enter a value...',
    value=res[7:-5],
    style={'width': '100%'}
),
	html.Hr(),
	html.Pre(id='hover-data', style=styles1['pre']),


    ])



@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(image):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open('/root/ImageCaptioning/data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(256).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(256, 512, len(vocab), 1)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load('/root/ImageCaptioning/models/encoder-5-3000.pkl', map_location='cpu'))
    decoder.load_state_dict(torch.load('/root/ImageCaptioning/models/decoder-5-3000.pkl', map_location='cpu'))

    encoder.eval()
    decoder.eval()
    # Prepare an image
    image = load_image(image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    return(sentence)

if __name__ == '__main__':

    app.run_server(debug=True)



