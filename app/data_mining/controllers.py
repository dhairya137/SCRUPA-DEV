from functools import wraps

from flask import abort, Blueprint, jsonify, request, send_from_directory, flash,redirect,url_for,session
from flask_login import current_user, login_user
from passlib.hash import sha256_crypt
from flask_login import login_required, current_user, login_user, logout_user

from app import data_loader, date_time_transformer, data_transformer, numerical_transformer, one_hot_encoder, \
    data_deduplicator, active_user_handler, UPLOAD_FOLDER,data_classifier,data_regression, data_reports
from app.history.models import History
from app.user_service.models import UserDataAccess

from flask import  render_template, make_response

from flask import Flask, render_template, request, redirect
import requests
import simplejson as json
import pandas as pd
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html, components
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral4
from bokeh.models import CategoricalColorMapper, Legend,LabelSet

from bokeh.io import output_file, show
from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource,
                          LinearColorMapper, PrintfTickFormatter,)
from bokeh.plotting import figure
from bokeh.sampledata.unemployment1948 import data
from bokeh.transform import transform
from bokeh.transform import dodge
from bokeh.models import Title


 

data_mining = Blueprint('data_mining', __name__)

def get_data(start_year, end_year, animal):
    # Loading the dataset on scientific procedures on animals in the UK
    #data = requests.get('https://www.quandl.com/api/v3/datasets/GUARDIAN/ANIMAL_TESTING.json?auth_token=APubRyAz2zP5ZWsNhDs2')
    #parsed_data = json.loads(data.text)
    # Making it a pandas DataFrame
    
    #df = pd.DataFrame(parsed_data)

    #df=pd.read_excel('results.xlsx')
    # Converting the dates into timestamps
    #for i in df['dataset']['data']:
    #   i[0] = pd.to_datetime(i[0])
    # Keeping only the data
    #plotting_data = df['dataset'][2:4]
    # Making a Bokeh-licious plot
    # First selecting the data
    years = ['2003','2004','2005']
    numbers = ['22222','22222222','2222']
    #index_of_animal = plotting_data['column_names'].index(animal)
     
    return [years, numbers]

def make_plot(years, numbers, message,algo):
    # The Data is ready for the plot

    
    source = ColumnDataSource(data=dict(years=list(years), numbers=list(numbers), color=Spectral4))
     
    f = figure(x_range=years,height = 400, width = 600)
    f.add_layout(Legend(), 'right')
    #f.legend.orientation = "horizontal"
    #f.legend.location = "top_center"

    f.vbar(x='years', top='numbers', width=0.9,color='color',legend_field="years", source=source)
    f.xgrid.grid_line_color = None
    f.y_range.start = 0
    labels = LabelSet(x='years', y='numbers', text='numbers', level='glyph',
        x_offset=-13.5, y_offset=0, source=source, render_mode='canvas')

    f.add_layout(labels)

    #f.line(years, numbers, color='#1F78B4')
    f.xaxis.axis_label = "Evaluation Measures"
    f.yaxis.axis_label = "Score"

    f.add_layout(Title(text=message, text_font_style="italic"), 'above')
    f.add_layout(Title(text=algo, text_font_size="16pt"), 'above')
    div,script = components(f, CDN)
    return [div, script]

def make_heatmap(fruits,data):
    # The Data is ready for the plot

 
    source = ColumnDataSource(data=data)
    print(source)

    color=["#c9d9d3","#718dbf","#e84d60","#effd60","#e8ff60","#c94d60","#714d60"]

    p = figure(x_range=fruits,height = 400, width = 600,  y_range=(0, 1.1), title="Correlation Matrix")
    
    ind=0

    offset=-0.25

    for fr in fruits:
        print(fr)
         

 
        p.vbar(x=dodge('fruits',offset,range=p.x_range), top=str(fr), width=0.2, source=source,
       color=color[ind], legend_label=fr)

        offset=offset+0.25

        ind=ind+1

    

    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.legend.location = "right"
    p.legend.orientation = "vertical"


    div,script = components(p, CDN)
    return [div, script]


 


@data_mining.route('/data_mining/visualize',methods=['GET'])
@login_required
def visualize():
     

    div=session['div']
    script=session['script']
    back_path=session['back_path']
     
    session.pop('div', None)
    session.pop('script', None)
    #session.pop('table', None)

     
    return render_template('data_mining/visualize.html', div = div, script = script,back_path=back_path)


 


@data_mining.route('/data_mining/datasets/<int:dataset_id>/tables/<string:table_name>/classifyNB', methods=['POST'])
@login_required
def classifyNB(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:

        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        test_ratio = request.args.get('test-ratio')
        results=data_classifier.naive_bayes(dataset_id, table_name, column_name,test_ratio)
        table = data_loader.get_table(dataset_id, table_name)
        
        #flash(u"data_mining/visualize.html", 'success') 
       
        numbers=results[0:4]

        numbers = [round(num, 2) for num in numbers]

        years=["Accuracy","Precision","Recall","F-score"]

        print(years)

        div, script = make_plot(years, numbers, results[4],results[5])
        #return render_template('data_mining/visualize.html', div = div, script = script)

        session['div']=div
        session['script']=script

        session['back_path']="/datasets/"+str(dataset_id)+"/tables/"+str(table_name)
        session['table_dataset']=dataset_id
        session['table_name']=table_name
        
 
     
        return (url_for('data_mining.visualize'))
       
 

    except Exception as error:
        

        flash(u"NB can not be applied "+str(error), 'danger')
        return jsonify({'error': True}), 400
@data_mining.route('/data_mining/datasets/<int:dataset_id>/tables/<string:table_name>/classifyKNN', methods=['PUT'])
@login_required
def classifyKNN(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        test_ratio = request.args.get('test-ratio')
        k = request.args.get('neighbours')
        results=data_classifier.k_nearest_neighbours(dataset_id, table_name, column_name,test_ratio,k)
        table = data_loader.get_table(dataset_id, table_name)
        numbers=results[0:4]
        numbers = [round(num, 2) for num in numbers]

        years=["Accuracy","Precision","Recall","F-score"]

        print(years)

        div, script = make_plot(years, numbers, results[4],results[5])
        #return render_template('data_mining/visualize.html', div = div, script = script)

        session['div']=div
        session['script']=script

        session['back_path']="/datasets/"+str(dataset_id)+"/tables/"+str(table_name)
        session['table_dataset']=dataset_id
        session['table_name']=table_name
        
 
     
        return (url_for('data_mining.visualize'))

    except Exception:
        flash(u"KNN can not be applied"+str(test_ratio), 'danger')
        return jsonify({'error': True}), 400



@data_mining.route('/data_mining/datasets/<int:dataset_id>/tables/<string:table_name>/classifyANN', methods=['PUT'])
@login_required
def classifyANN(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        test_ratio = request.args.get('test-ratio')
        criterion = request.args.get('criterion')
        results=data_classifier.artificial_neural_net(dataset_id, table_name, column_name,test_ratio,criterion)
        table = data_loader.get_table(dataset_id, table_name)
        numbers=results[0:4]
        numbers = [round(num, 2) for num in numbers]

        years=["Accuracy","Precision","Recall","F-score"]

        print(years)

        div, script = make_plot(years, numbers, results[4],results[5])
        #return render_template('data_mining/visualize.html', div = div, script = script)

        session['div']=div
        session['script']=script

        session['back_path']="/datasets/"+str(dataset_id)+"/tables/"+str(table_name)
        session['table_dataset']=dataset_id
        session['table_name']=table_name
        
 
     
        return (url_for('data_mining.visualize'))

    except Exception:
        flash(u"DT can not be applied"+str(test_ratio), 'danger')
        return jsonify({'error': True}), 400



@data_mining.route('/data_mining/datasets/<int:dataset_id>/tables/<string:table_name>/classifyDT', methods=['PUT'])
@login_required
def classifyDT(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        test_ratio = request.args.get('test-ratio')
        criterion = request.args.get('criterion')
        results=data_classifier.decision_tree(dataset_id, table_name, column_name,test_ratio,criterion)
        table = data_loader.get_table(dataset_id, table_name)
        numbers=results[0:4]
        numbers = [round(num, 2) for num in numbers]

        years=["Accuracy","Precision","Recall","F-score"]

        print(years)

        div, script = make_plot(years, numbers, results[4],results[5])
        #return render_template('data_mining/visualize.html', div = div, script = script)

        session['div']=div
        session['script']=script

        session['back_path']="/datasets/"+str(dataset_id)+"/tables/"+str(table_name)
        session['table_dataset']=dataset_id
        session['table_name']=table_name
        
 
     
        return (url_for('data_mining.visualize'))



    except Exception:
        flash(u"DT can not be applied"+str(test_ratio), 'danger')
        return jsonify({'error': True}), 400


#http://localhost:5000/data_mining/0/tables/Iris/cmatrix
#http://localhost:5000/data_mining/0/tables/Iris/cmatrix
@data_mining.route('/data_mining/<int:dataset_id>/tables/<string:table_name>/cmatrix', methods=['GET'])
@login_required
def cmatrix(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)

        print('here')

         

        results=data_reports.correlation_matrix(dataset_id, table_name)

        years=results.columns.tolist()
        fruits=years
         
       
        data={'fruits':fruits}

        for ind,row in results.iterrows():
            data[ind]=row.tolist()


        
        div, script = make_heatmap(fruits,data)
        #return render_template('data_mining/visualize.html', div = div, script = script)

         
        back_path="/datasets/"+str(dataset_id)+"/tables/"+str(table_name)

        
 
     
        #return (url_for('data_mining.visualize'))
        return render_template('data_mining/visualize.html', div = div, script = script,back_path=back_path,table_dataset=dataset_id,table_name=table_name)
       
 

    except Exception as error:
        print('lalalalalalalalaalalalalall')

        flash(u"NB can not be applied "+str(error), 'danger')
        return jsonify({'error': True}), 400