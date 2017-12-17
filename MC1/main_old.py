import string
import time
from datetime import datetime, date, timedelta
import random
from math import sin, cos, radians, pi
import itertools
import json

import operator
from collections import defaultdict
import warnings

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, PreText, Slider, Button, DateRangeSlider
from bokeh.layouts import widgetbox, column, layout, row
from bokeh.models import HoverTool, BoxZoomTool, ResetTool, LassoSelectTool, \
    TextInput, Arrow, OpenHead, NormalHead, VeeHead, BoxSelectTool, Span, PanTool
from bokeh.plotting import figure
from bokeh.models.widgets import RadioGroup

from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.snowball import SnowballStemmer

from itertools import chain

import preprocessor as p

import numpy as np
import pandas as pd

from gensim.models import Word2Vec

import get_filtered_tweets

flag_play = False

# Ignore warnings
warnings.filterwarnings('ignore')

dict_index = {}

p.set_options(p.OPT.EMOJI,p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.SMILEY)
stemmer = SnowballStemmer("english")

file_tweets = 'HW7/static/Microblogs_processed.txt'
file_w2v = 'HW7/static/w2v.model'
file_w2v_index = 'HW7/static/word_tweetIndices.txt'
file_weather = 'HW7/static/Weather.csv'
file_image_vastopolis = 'HW7/static/Vastopolis_Map.png'

# file_tweets = './static/Microblogs_processed.txt'
# file_w2v = './static/w2v.model'
# file_w2v_index = './static/word_tweetIndices.txt'
# file_weather = './static/Weather.csv'
# file_image_vastopolis = './static/Vastopolis_Map.png'

max_cloud_count = 25

def update_dt_line(curr):
    ts_curr.location = (curr - datetime(1970, 1, 1)).total_seconds()*1000

#callback when the play button is pressed
def update_visualization():
    global curr_dt
    global flag_play, relevant_tweet

    tweet_subset = relevant_tweet[(relevant_tweet.Created_at < curr_dt)]

    update_dt_line(curr_dt)
    plot_scatter(tweet_subset)
    plot_wind(curr_dt.date())

    date_string = curr_dt.strftime('%Y-%m-%d %H:%M:%S')
    dur_text.text = date_string

    if curr_dt == end_dt:
        curr_dt = start
        doc.remove_periodic_callback(update_visualization)
        flag_play = False
        button_play.label = 'Play'
        date_range_slider.disabled = False


    x = sldr_rate.value
    curr_dt = curr_dt + timedelta(hours=x)

    if curr_dt > end_dt:
        curr_dt = end_dt


def plot_scatter(relevant_tweet_df):
    src.data['Long'] = relevant_tweet_df['Long'].tolist()
    src.data['Lat'] = relevant_tweet_df['Lat'].tolist()
    src.data['tweet_text'] = relevant_tweet_df['text'].tolist()
    src.data['Created_at'] = relevant_tweet_df['Created_at'].tolist()
    src.data['Color'] = relevant_tweet_df['Color'].tolist()
    plt_src.data_source.trigger('data', plt_src.data_source.data, plt_src.data_source.data)

#callback when play button is pressed
def bt_play_click():
    global flag_play, curr_dt, end_dt

    if flag_play is False:
        val = date_range_slider.value
        print(val)

        if(isinstance(val[0], date)):
            curr_dt = datetime.combine(val[0], datetime.min.time())
            end_dt = datetime.combine(val[1], datetime.min.time())
            # curr_dt = datetime.fromtimestamp(val[0])
            # end_dt = datetime.fromtimestamp(val[1])
        else:
            curr_dt = datetime.fromtimestamp(val[0] / 1000)
            end_dt = datetime.fromtimestamp(val[1] / 1000)

        doc.add_periodic_callback(update_visualization, 4000)
        button_play.label = 'Pause'
        flag_play = True
        date_range_slider.disabled = True
    else:
        doc.remove_periodic_callback(update_visualization)
        button_play.label = 'Play'
        flag_play = False
        date_range_slider.disabled = False

def import_index():
    with open(file_w2v_index) as tweetfile:
        index = json.load(tweetfile)
    return index

def get_trend(relevant_tweets):
    df = relevant_tweets.groupby(
        [relevant_tweets['Created_at'].apply(lambda x: x.replace(second=0, minute=0) + timedelta(hours=1))]).size()
    dates = list(df.index)
    counts = list(df.values)

    idx = relevant_tweets.groupby(['ID'])['Created_at'].transform(min) == relevant_tweets['Created_at']

    df1 = relevant_tweets[idx].groupby(
        [relevant_tweets['Created_at'].apply(lambda x: x.replace(second=0, minute=0) + timedelta(hours=1))]).size()

    dates1 = list(df1.index)
    counts1 = list(df1.values)

    indices = relevant_tweet[idx].index

    return dates, counts, dates1, counts1, indices

def generate_cloud_data(relevant_words):
    word = []
    font_size = []
    if len(relevant_words.keys()) > max_cloud_count:
        dic = dict(sorted(relevant_words.items(), key=operator.itemgetter(1), reverse=True)[:max_cloud_count])
    else:
        dic = dict(relevant_words)

    top_count = len(dic.keys())
    x_rand = random.sample(range(-10, 110), top_count)
    y_rand = random.sample(range(10, 90), top_count)

    s = sum(list(dic.values()))
    if s is not 0:
        for key, value in dic.items():
            word.append(key)
            val = value/s * 200

            if (val < 3):
                val = 3
            if (val > 20):
                val = 20
            font_size.append("{0:.2f}".format(val) + 'pt')

    return x_rand, y_rand, word, font_size


def plot_word_cloud(relevant_words):
    global source_cloud_1
    x_rand, y_rand, word, font_size = generate_cloud_data(relevant_words)

    source_cloud_1.data['x'] = x_rand
    source_cloud_1.data['y'] = y_rand
    source_cloud_1.data['font_size'] = font_size
    source_cloud_1.data['word'] = word


def plot_event_cloud(relevant_words):
    global source_cloud_2

    x_rand, y_rand, word, font_size = generate_cloud_data(relevant_words)

    source_cloud_2.data['x'] = x_rand
    source_cloud_2.data['y'] = y_rand
    source_cloud_2.data['font_size'] = font_size
    source_cloud_2.data['word'] = word


def plot_trend_graph(dates, counts, dates1, counts1):
    line1_datasource.data['x'] = dates
    line1_datasource.data['y'] = counts
    line1_datasource.trigger('data', line1_datasource.data, line1_datasource.data)

    line2_datasource.data['x'] = dates1
    line2_datasource.data['y'] = counts1
    line2_datasource.trigger('data', line2_datasource.data, line2_datasource.data)


def get_tweet_word2vec(symptoms, tweet_subset):
    relevant_words = defaultdict(int)
    if len(symptoms) is 0:
        return relevant_words, tweet_subset

    if len(tweet_subset) > 200000:
        print("Too big to generate model")
        new_model = model
        print("MODEL not updated")
    else:
        new_model = get_filtered_tweets.update_word2vec_model(tweet_subset.Words.apply(lambda Word: Word.split(',')).tolist())

    catch_words = [x.strip() for x in symptoms.split(',')]
    catch_words = [stemmer.stem(word) for word in catch_words]

    for word in catch_words:
        if word not in new_model.wv.vocab:
            continue
        if word not in relevant_words.keys():
            relevant_words[word] = 1
        similarWords = new_model.most_similar(word, topn=20)
        #print(similarWords)
        for t in similarWords:
            if t[1] > 0.75 and t[0] not in relevant_words.keys():
                relevant_words[t[0]] = t[1]
        relevant_words = {k: v for k, v in relevant_words.items() if v != 0}
    print(relevant_words)
    index = set()
    for word in relevant_words.keys():
        if word in dict_index.keys():
            index.update(dict_index[word])

    ind = tweet_subset.index.intersection(list(index))

    rel_tweet = tweet_subset.loc[ind, :]
    return relevant_words, rel_tweet


def update_sliders(df_tweet):
    start = df_tweet.Created_at.min()
    end = df_tweet.Created_at.max()

    t1 = end - start
    y1 = int((t1.total_seconds() / 3600) / 10)
    y2 = int((t1.total_seconds() / 3600) / 4)

    sldr_rate.end = y2
    sldr_rate.value = y1

    print(y1, y2)


def bt_compare_click():
    user_text.text = "Please wait ....."
    global relevant_tweet
    # use Word2vec
    val = date_range_slider.value
    start_date = datetime.fromtimestamp(val[0] / 1000)
    end_date = datetime.fromtimestamp(val[1] / 1000)
    print(start_date)
    print(end_date)
    relevant_tweet = tweet_dataset[(tweet_dataset.Created_at > start_date) & (tweet_dataset.Created_at < end_date)]
    print("Time filter", relevant_tweet.shape)
    relevantWords, relevant_tweet = get_tweet_word2vec(search_1.value, relevant_tweet)
    print("Word2vec filter", relevant_tweet.shape)

    relevant_tweet['Color'] = 'yellow'
    dates, counts, dates1, counts1, idx = get_trend(relevant_tweet)
    relevant_tweet.loc[list(idx), 'Color'] = 'red'
    plot_word_cloud(relevantWords)
    plot_event_cloud(dict())
    plot_trend_graph(dates, counts, dates1, counts1)
    plot_scatter(relevant_tweet)
    update_sliders(relevant_tweet)
    user_text.text = ""


def plot_initial_vis():
    global relevant_tweet

    relevant_tweet['Color'] = 'yellow'
    relevantWords, relevant_tweet = get_tweet_word2vec(default_search_1, relevant_tweet)

    dates, counts, dates1, counts1, idx = get_trend(relevant_tweet)
    relevant_tweet.loc[list(idx), 'Color'] = 'red'

    print("Initial count", relevant_tweet.shape)

    plot_scatter(relevant_tweet)
    plot_word_cloud(relevantWords)
    plot_trend_graph(dates, counts, dates1, counts1)


def point_pos(x0, y0, d, theta):
    theta_rad = radians(theta)
    return x0 - d*cos(theta_rad), y0 + d*sin(theta_rad)

def read_weather_file():
    data1 = pd.read_csv(file_weather, error_bad_lines=False, delimiter=",")
    data1 = data1.dropna()
    data1['Date'] = pd.to_datetime(data1['Date'], errors='coerce', infer_datetime_format=True)
    data1['Weather'] = data1['Weather'].astype(str)
    data1['Average_Wind_Speed'] = data1['Average_Wind_Speed'].astype(int)
    data1['Angle'] = data1['Angle'].astype(int)
    print(data1.shape)
    return data1

def plot_wind(date):
    px_start = 93.1923
    px_end = 93.5673
    py_start = 42.1608
    py_end = 42.3017
    x_start = np.arange(px_start, px_end, (px_end - px_start)/10)
    y_start = np.arange(py_start, py_end, (py_end - py_start)/5)

    curr_weather = weather_data[weather_data['Date'] == date]

    if curr_weather.empty:
        return

    wind_dir = curr_weather['Angle']

    x_end = []
    y_end = []

    start = list(itertools.product(x_start, y_start))

    x_start = []
    y_start = []

    for x in start:
        x1, y1 = point_pos(x[0], x[1], 0.01, wind_dir)
        x_end.append(x1)
        y_end.append(y1)
        x_start.append(x[0])
        y_start.append(x[1])

    source_wind.data['x1'] = x_start
    source_wind.data['y1'] = y_start
    source_wind.data['x2'] = x_end
    source_wind.data['y2'] = y_end
    arrows.source.trigger('data', arrows.source.data, arrows.source.data)


def on_selection_change(attr, old, new):
    user_text.text = "Please wait ....."
    inds = new['1d']['indices']
    #print(new)
    if len(inds) != 0:
        tweet_subset = relevant_tweet.iloc[inds, :]
        events = get_filtered_tweets.get_events(tweet_subset)
        plot_event_cloud(events)
        print(events)

    user_text.text = ""

########################get Dataset

model = Word2Vec.load(file_w2v)
dict_index = import_index()
tweet_dataset = get_filtered_tweets.get_preprocessed_data()
weather_data = read_weather_file()
relevant_tweet = tweet_dataset.copy(deep=True)
# datetime_object = datetime.strptime('2011-05-02 10:00:00', "%Y-%m-%d %H:%M:%S")
# plot_wind(datetime_object.date())
#tweet_dataset['text_tokenized'] = tweet_dataset.Words.apply(lambda Word: Word.split(','))
default_search_1 = "flu,sick,fever,aches,pains,fatigue,coughing,vomiting,diarrhea"
search_term_1 = default_search_1
search_1 = TextInput(value=default_search_1, title="Enter Keywords:")
sldr_w2v_ = Slider(start=1, end=30, value=20, step=1, title="Maximum number of similar words")

# start = datetime.strptime('2011-05-16 10:00:00', "%Y-%m-%d %H:%M:%S")
# end = datetime.strptime('2011-05-18 10:00:00', "%Y-%m-%d %H:%M:%S")
#
# relevant_tweet = relevant_tweet[(tweet_dataset.Created_at > start) & (tweet_dataset.Created_at < end)]
# use Word2vec

# ########### Create Visualizations ##################
#
# Line graph for trend
hover1 = HoverTool(tooltips=[
    ("Count", "@y"),
    ("Time", "@x")
])

src_line_1 = ColumnDataSource(data=dict(x=[], y=[]))

plot_trend = figure(title='Trend of Tweets', plot_width=500, plot_height=250, x_axis_type="datetime",
                    tools=[hover1, ResetTool(), BoxZoomTool()])
line1 = plot_trend.line(x='x', y='y', source=src_line_1, line_width=2, line_color='red', legend='Previous users')
line1_datasource = line1.data_source
line2 = plot_trend.line(x=[], y=[], line_width=2, line_color='Yellow', legend='New users')
line2_datasource = line2.data_source
plot_trend.legend.location = "top_left"
plot_trend.toolbar.logo = None
plot_trend.xaxis.axis_label = "Date"
plot_trend.yaxis.axis_label = "Relevant Tweet Count"
plot_trend.min_border = 0

ts_curr = Span(location=0,dimension='height', line_color='green', line_dash='dashed', line_width=3)
plot_trend.renderers.extend([ts_curr])

# Widgets - Search, button
button_go = Button(label="Search Revelant Tweets", width=50, button_type="success")
button_go.on_click(bt_compare_click)

user_text = PreText(text="")

################################

src = ColumnDataSource(data=dict(Lat=[], Long=[], Created_at=[], tweet_text=[], Color=[]))

hover2 = HoverTool(tooltips=[
    ("Tweet", "@tweet_text")
])

# init figure
plot_tweet = figure(title="Origin of Tweets", toolbar_location="above",
                    plot_width=931, plot_height=350, x_range=(93.5673, 93.1923), y_range=(42.1409, 42.3117),
                    tools=[hover2, LassoSelectTool(), BoxSelectTool(), ResetTool(), BoxZoomTool(), PanTool()])

plot_tweet.image_url(url=[file_image_vastopolis], x=93.5673, y=42.1609, w=0.375, h=0.1509, anchor="bottom_left")
plt_src = plot_tweet.circle(x='Long', y='Lat', alpha=0.2, size=2, color='Color', source=src)
plot_tweet.axis.visible = False
plot_tweet.toolbar.logo = None
plot_tweet.xgrid.grid_line_color = None
plot_tweet.ygrid.grid_line_color = None
plot_tweet.min_border = 0
plt_src.data_source.on_change('selected', on_selection_change)


#plot_wind()

###########################################################
source_wind = ColumnDataSource(data=dict(
        x1=[], y1=[], x2=[], y2=[]))

arrows = Arrow(end=VeeHead(fill_color="black", size=10, fill_alpha=.8, line_alpha=.5),
               x_start='x1', y_start='y1', x_end='x2', y_end='y2', source=source_wind)

plot_tweet.add_layout(arrows)

##################### slider, play buttons
start = relevant_tweet.Created_at.min() - timedelta(days=1)
end = relevant_tweet.Created_at.max() + timedelta(days=1)

curr_dt = start
end_dt = end
print(start, end, curr_dt)

plot_wind(curr_dt.date())

t1 = end - start
y1 = int((t1.total_seconds()/3600) / 10)
y2 = int((t1.total_seconds()/3600) / 4)
sldr_rate = Slider(start=1, end=y2, value=y1, step=1, title="Rate of change (in hours)")


date_range_slider = DateRangeSlider(title="Date Range: ", start=start.date(), end=end.date(), value=(start.date(), end.date()), step=10000)
#date_range_slider.on_change('value', cb_sldr_time)

button_play = Button(label="Play", button_type="success")
button_play.on_click(bt_play_click)

dur_text = PreText(text="", height=50)

######################## Word clouds for most relevant words
word_cloud_stream_1 = figure(x_range=(-20, 150), y_range=(0, 100),plot_width=300, plot_height=250, tools=[BoxZoomTool(), ResetTool()], title="Related Words")
word_cloud_stream_1.toolbar.logo = None
word_cloud_stream_1.axis.visible = False
word_cloud_stream_1.xgrid.grid_line_color = None
word_cloud_stream_1.ygrid.grid_line_color = None

source_cloud_1 = ColumnDataSource(data=dict(x=[], y=[], word=[], font_size=[]))
word_cloud_stream_1.text(x='x', y='y', text='word', text_font_size='font_size', source=source_cloud_1)

word_cloud_stream_2 = figure(x_range=(-20, 150), y_range=(0, 100), plot_width=300, plot_height=250,
                             tools=[BoxZoomTool(), ResetTool()], title="Related Events")
word_cloud_stream_2.toolbar.logo = None
word_cloud_stream_2.axis.visible = False
word_cloud_stream_2.xgrid.grid_line_color = None
word_cloud_stream_2.ygrid.grid_line_color = None

source_cloud_2 = ColumnDataSource(data=dict(x=[], y=[], word=[], font_size=[]))
word_cloud_stream_2.text(x='x', y='y', text='word', text_font_size='font_size', source=source_cloud_2)

plot_initial_vis()

wgt_but = column(widgetbox(button_go))
wgt_txt = column( widgetbox(user_text))
wgt_search = row(children=[widgetbox(search_1)])
wgt_media_1 = column(widgetbox(button_play), widgetbox(dur_text), widgetbox(sldr_rate))

doc = curdoc()

layout = layout(children=[[wgt_search, date_range_slider, wgt_but, wgt_txt], [plot_tweet, wgt_media_1],
                          [plot_trend, word_cloud_stream_1, word_cloud_stream_2]])

doc.add_root(layout)