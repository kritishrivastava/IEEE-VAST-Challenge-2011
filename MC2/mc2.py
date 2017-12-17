import pandas as pd
import os
import csv
from collections import defaultdict
from bokeh.io import show
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
)
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.layouts import widgetbox, column, layout, row
from bokeh.models.widgets import Select
import pickle
import xml.etree.ElementTree
from xml.etree import ElementTree


def process_IDS_data():
    """
    Read and preprocess IDS data
    :return: list of preprocessed IDS entries and count dataframe dumps
    """
    processed_ids_entries = []
    paths = []
    path = os.getcwd()
    paths.append(path + "\MiniChallenge2 Core Data\\20110413\IDS\\20110413_VAST11MC2_IDS.txt")
    paths.append(path + "\MiniChallenge2 Core Data\\20110414\IDS\\20110414_VAST11MC2_IDS.txt")
    paths.append(path + "\MiniChallenge2 Core Data\\20110415\IDS\\20110415_VAST11MC2_IDS.txt")
    for path in paths:
        ids_log = open(path, "r").read().split("\n\n")
        # write_path = os.getcwd()
        # write_path = write_path + "\MiniChallenge2 Core Data\\20110413\IDS\\20110413_VAST11MC2_IDS.csv"
        # writer = csv.writer(open(write_path, 'w', newline=""))
        # header = ['datetime', 'date', 'hour', 'min', 'type', 'error','priority','source','destination']
        # writer.writerow(header)
        for entry in ids_log:
            lines = entry.splitlines()
            if len(lines) > 0:
                # Process line 0 - get type and error
                s = lines[0]
                type = s[s.find("(") + 1:s.find(")")]
                second_half = s.split(")")
                if len(second_half) >= 2:
                    second_half = second_half[1]
                    if second_half[1] == ':':
                        error = second_half[3:second_half.find("[")-1]
                    else:
                        error = second_half[1:second_half.find("[")-1]
                # Process line 1 - get priority
                s = lines[1]
                priority = s[s.find(":")+2:s.find("]")]
                # Process line 2 - get date, hour, min, source and destination
                s = lines[2]
                date = s[3:5]
                hour = s[6:8]
                min = s[9:11]
                parts = s.split(" ")
                datetime = parts[0]
                source = parts[1]
                destination = parts[3]
                # Write entry to file
                # writer.writerow([datetime, date, hour, min, type, error, priority, source, destination])
                processed_ids_entries.append([datetime, date, hour, min, type, error, priority, source, destination])
    with open('processed_ids_entries.pkl', 'wb') as f:
        pickle.dump(processed_ids_entries, f)
    f.close()
    ids_count = defaultdict(int)
    for entry in processed_ids_entries:
        ids_count[str(entry[1]) + ":" + (entry[2]) + ":" + str(entry[3])] += 1
    with open('ids_count.pickle', 'wb') as handle:
        pickle.dump(ids_count, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    dates = []
    hours = []
    mins = []
    for time in ids_count.keys():
        time = time.split(":")
        dates.append(str(time[0]) + '/Apr/2011')
        hours.append(time[1])
        mins.append(time[2])
    d = {'date': dates, 'hour': hours, 'min': mins, 'count': list(ids_count.values())}
    ids_df = pd.DataFrame(data=d)
    ids_df.to_pickle("ids_df.pkl")

def process_firewall_data():
    """
    read firewall date and get the counts
    :return: count of entries for every hour:min and dataframe dumps
    """
    firewall_count = defaultdict(int)
    path = os.getcwd()
    paths = []
    paths.append(path + "\MiniChallenge2 Core Data\\20110413\\firewall\csv")
    paths.append(path + "\MiniChallenge2 Core Data\\20110414\\firewall\csv")
    paths.append(path + "\MiniChallenge2 Core Data\\20110415\\firewall\csv")
    for path in paths:
        for filename in os.listdir(path):
            filepath = path + "\\" + filename
            with open(filepath, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                for i, line in enumerate(reader):
                    if i != 0:
                        if len(line) >= 1:
                            datetime = line[0].split(",")[0].split(" ")
                            date = datetime[0]
                            time = datetime[1]
                            firewall_count[date + ":" + time[:5]] +=1
            f.close()
    with open('firewall_count.pickle', 'wb') as handle:
        pickle.dump(firewall_count, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    dates = []
    hours = []
    mins = []
    for time in firewall_count.keys():
        time = time.split(":")
        dates.append(time[0])
        hours.append(time[1])
        mins.append(time[2])
    d = {'date': dates, 'hour': hours, 'min': mins, 'count': list(firewall_count.values())}
    firewall_df = pd.DataFrame(data=d)
    firewall_df.to_pickle("firewall_df.pkl")

def process_security_data():
    """
    Read and preprocess security xml data
    :return: count and dataframe dumps
    """
    path = os.getcwd()
    paths = []
    paths.append(path + "\MiniChallenge2 Core Data\\20110413\security\\20110413_VAST11MC2_SecurityLog.xml")
    paths.append(path + "\MiniChallenge2 Core Data\\20110414\security\\20110414_VAST11MC2_SecurityLog.xml")
    paths.append(path + "\MiniChallenge2 Core Data\\20110415\security\\20110415_VAST11MC2_SecurityLog.xml")
    security_count = defaultdict(int)
    security_event = defaultdict(int)
    security_time = []
    security_eventid = []
    for path in paths:
        xml.etree.ElementTree.register_namespace("lc", 'http://schemas.microsoft.com/win/2004/08/events/event')
        tree = xml.etree.ElementTree.parse(path)
        for node in tree.iter():
            if(node.tag == '{http://schemas.microsoft.com/win/2004/08/events/event}TimeCreated'):
                systemtime = node.attrib['SystemTime']
                date = systemtime[8:10]
                time = systemtime[11:16]
                security_count[str(date)+ ':' + str(time)] +=1
                security_time.append(systemtime)
            if(node.tag == '{http://schemas.microsoft.com/win/2004/08/events/event}EventID'):
                security_eventid.append(node.text)
                security_event[node.text] +=1
    print(security_event)
    # print(security_eventid)
    #exit()
    with open('security_count.pickle', 'wb') as handle:
        pickle.dump(security_count, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    dates = []
    hours = []
    mins = []
    for time in security_count.keys():
        time = time.split(":")
        dates.append(str(time[0]) + '/Apr/2011')
        hours.append(time[1])
        mins.append(time[2])
    d = {'date': dates, 'hour': hours, 'min': mins, 'count': list(security_count.values())}
    security_df = pd.DataFrame(data=d)
    security_df.to_pickle("security_df.pkl")


def get_preprocessed_data():
    """
    read all pickle files
    :return: IDS, firewall, security dataframes and count lists
    """
    ids_df = pd.read_pickle("ids_df.pkl")
    with open('ids_count.pickle', 'rb') as handle:
        ids_count = pickle.load(handle)
    handle.close()
    firewall_df = pd.read_pickle("firewall_df.pkl")
    with open('firewall_count.pickle', 'rb') as handle:
        firewall_count = pickle.load(handle)
    handle.close()
    security_df = pd.read_pickle("security_df.pkl")
    with open('security_count.pickle', 'rb') as handle:
        security_count = pickle.load(handle)
    handle.close()
    return ids_df, ids_count, firewall_df, firewall_count, security_df, security_count


def change_date(attr, old, new):
    """
    update heatmaps
    :param new: value of the date selected from dropdown
    :return: none
    """
    global firewall_heatmap, ids_heatmap, security_heatmap
    if new == "April 13, 2011":
        firewall_df_src = firewall_df.loc[firewall_df['date'] == '13/Apr/2011']
        ids_df_src = ids_df.loc[ids_df['date'] == '13/Apr/2011']
        security_df_src = security_df.loc[security_df['date'] == '13/Apr/2011']
    elif new == "April 14, 2011":
        firewall_df_src = firewall_df.loc[firewall_df['date'] == '14/Apr/2011']
        ids_df_src = ids_df.loc[ids_df['date'] == '14/Apr/2011']
        security_df_src = security_df.loc[security_df['date'] == '14/Apr/2011']
    else:
        firewall_df_src = firewall_df.loc[firewall_df['date'] == '15/Apr/2011']
        ids_df_src = ids_df.loc[ids_df['date'] == '15/Apr/2011']
        security_df_src = security_df.loc[security_df['date'] == '15/Apr/2011']
    firewall_heatmap.data_source.data['date'] = firewall_df_src['date'].tolist()
    firewall_heatmap.data_source.data['hour'] = firewall_df_src['hour'].tolist()
    firewall_heatmap.data_source.data['min'] = firewall_df_src['min'].tolist()
    firewall_heatmap.data_source.data['count'] = firewall_df_src['count'].tolist()
    firewall_heatmap.data_source.trigger('data', firewall_heatmap.data_source.data, firewall_heatmap.data_source.data)
    ids_heatmap.data_source.data['date'] = ids_df_src['date'].tolist()
    ids_heatmap.data_source.data['hour'] = ids_df_src['hour'].tolist()
    ids_heatmap.data_source.data['min'] = ids_df_src['min'].tolist()
    ids_heatmap.data_source.data['count'] = ids_df_src['count'].tolist()
    ids_heatmap.data_source.trigger('data', ids_heatmap.data_source.data, ids_heatmap.data_source.data)
    security_heatmap.data_source.data['date'] = security_df_src['date'].tolist()
    security_heatmap.data_source.data['hour'] = security_df_src['hour'].tolist()
    security_heatmap.data_source.data['min'] = security_df_src['min'].tolist()
    security_heatmap.data_source.data['count'] = security_df_src['count'].tolist()
    security_heatmap.data_source.trigger('data', security_heatmap.data_source.data, security_heatmap.data_source.data)

def create_heatmap(df, colors, title):
    """
    :param df: dataframe with time and count data
    :param colors: color palette
    :param title: title of the plot - firewall/IDS/security
    :return: bokeh heatmap and figure variables
    """
    df_src = df.loc[df['date'] == '13/Apr/2011']
    source = ColumnDataSource(data = df_src)
    mapper = LinearColorMapper(palette=colors, low=min(df_src['count'].tolist()), high=max(df_src['count'].tolist()))
    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
    p = figure(title=title, x_axis_location="above", plot_width=1000, plot_height=200, tools=TOOLS, toolbar_location='right')
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = 3.14 / 3
		p.xaxis.axis_label = "Minutes"
		p.yaxis.axis_label = "Hours"
    heatmap = p.rect(x="min", y="hour", width=1, height=1, source=source, fill_color={'field': 'count', 'transform': mapper}, line_color=None)
    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%d"),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')
    p.select_one(HoverTool).tooltips = [
        ('Time', '@hour @min'),
        ('Count', '@count'),
    ]
    return p, heatmap


# Execute once --- Preprocess data #
# process_firewall_data()
# process_IDS_data()
#	process_security_data()
# Read preprocessed data #
ids_df, ids_count, firewall_df, firewall_count, security_df, security_count = get_preprocessed_data()
# Plot graphs #
## Firewall
firewall_colors = ["#ffe5e5", "#ffcccc", "#ffb2b2", "#ff9999", "#ff7f7f", "#ff6666", "#ff4c4c", "#ff3232", "#ff1919", "#ff0000"]
firewall_fig, firewall_heatmap = create_heatmap(firewall_df, firewall_colors, "Firewall")
## IDS
ids_colors = ["#f9e79f", "#f8c471", "#eb984e", "#dc7633", "#d35400"]
ids_fig, ids_heatmap = create_heatmap(ids_df, ids_colors, "IDS")
## Security
ids_colors = ["#f9e79f", "#f8c471", "#eb984e", "#dc7633", "#d35400"]
security_fig, security_heatmap = create_heatmap(security_df, ids_colors, "Security")
## Date dropdown
select = Select(title="Select Date:", value="April 13, 2011", options=["April 13, 2011", "April 14, 2011", "April 15, 2011"])
select.on_change('value', change_date)
wgt_search = row(children=[widgetbox(select)])
doc = curdoc()
layout1 = layout(children=[[wgt_search], [firewall_fig], [ids_fig], [security_fig]])
doc.add_root(layout1)