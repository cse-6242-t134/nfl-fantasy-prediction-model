import bokeh
import numpy as np
import pandas as pd
import math
import random

from bokeh.models import Button,PreText,ColumnDataSource, MultiSelect, CustomJS, DataTable, TableColumn, Div, TabPanel, Tabs, Tooltip, FactorRange, Select, TextInput
from bokeh.plotting import figure, show, row, column
from bokeh.io import save, show, output_file
from bokeh.events import ButtonClick
from bokeh import palettes

def visualize(datam = None, viz = None):
    src = pd.read_csv(datam)
    src.drop("Unnamed: 0", axis=1, inplace=True) if "Unnamed: 0" in src.columns else None
    output_file(viz, title = "Team 134: Fantasy Points Visualization")
    src = src[src['season'] == 2024]
    src[['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds', 'receptions', 'interceptions']] = src[['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds', 'receptions', 'interceptions']].fillna(0)
    src['weeks'] = 'Week ' + src['week'].astype(str)
    Viridis = palettes.Viridis256
    colors1 = ['blue'] * len(src)
    colors2 = ['red'] * len(src)
    src['colors1'] = colors1
    src['colors2'] = colors2
    src['colors3'] = random.choices(Viridis, k=len(src))
    src['predicted_fantasy'] = src['predicted_fantasy'].astype(float)
    src['predicted_fantasy'] = src['predicted_fantasy'].round(1)
    src['fantasy_points_ppr'] = src['fantasy_points_ppr'].round(1)
    src = src.sort_values(by='predicted_fantasy', ascending=False)
    dest = src.copy()
    dest.loc[:, ['predicted_fantasy', 'fantasy_points_ppr','colors1', 'colors2']] = '' 
    source = ColumnDataSource(src)
    srctarget = dest
    src2 = src.head(32)
    dest2 = dest.head(32)
    stateChange= ColumnDataSource(srctarget)
    sc2 = ColumnDataSource(dest2)
    source2 = ColumnDataSource(src2)
    div2 = Div(text="""<h2 style="font-size:15px; color:black;">Data Table for Scatter Plots</h2>""",
                            sizing_mode="fixed", height=24, width=200)
    div3 = Div(text="""<h2 style="font-size:15px; color:black;">Data Table for Bar Plots</h2>""",
                            sizing_mode="fixed", height=24, width=200)

    TOOLS = "hover,pan,wheel_zoom,box_zoom,reset,save"

    code_js = """ 
        function update() {
            var s1= sel1.value;
            var s2= sel2.options;
            for (const key in sc.data) {
                sc.data[key] = [];
            }
            for (const key in sc2.data) {
                sc2.data[key] = [];
            }
            if (s1 === undefined || s1.length == 0) {
                pr.text = "Please select a week";
                return;
            }
            pr.text = "";
            var lo = 0.0;
            var lo2 = 0.0;
            const pl = [];
            const pl2 = [];
            for (var i = 0; i < source.get_length(); i++) {
                if (s1 === 'All') {
                    if (!(s2 === undefined || s2.length == 0)) {
                        if(s2.indexOf(source.data['player_name'][i]) >= 0) { 
                            if(sc.data['index'].indexOf(source.data['index'][i]) < 0) {
                                for (const col in source.data) {
                                    sc.data[col].push(source.data[col][i]);                
                                }
                            }
                            if(sc2.get_length() < 32 && sc2.data['index'].indexOf(source.data['index'][i]) < 0) {
                                for (const col in source.data) {
                                    sc2.data[col].push(source.data[col][i]);                
                                }
                            }
                        }
                        else {
                            if(sc.data['index'].indexOf(source.data['index'][i]) >= 0) {
                                var q = sc.data['index'].indexOf(source.data['index'][i]);
                                for (const col in source.data) {
                                    sc.data[col].splice(q, 1);                
                                }
                                
                            }
                            if(sc2.data['index'].indexOf(source.data['index'][i]) >= 0) {
                                var q = sc2.data['index'].indexOf(source.data['index'][i]);
                                for (const col in source.data) {
                                    sc2.data[col].splice(q, 1);                
                                }
                                
                            }
                        }
                    }
                    else {
                        if(sc.data['index'].indexOf(source.data['index'][i]) < 0) {
                            pl.push(source.data['player_name'][i]);
                            for (const col in source.data) {
                                sc.data[col].push(source.data[col][i]);                
                            }
                        }
                        if(sc2.get_length() < 32 && sc2.data['index'].indexOf(source.data['index'][i]) < 0) {
                            pl2.push(source.data['player_name'][i]);
                            for (const col in source.data) {
                                sc2.data[col].push(source.data[col][i]);                
                            }
                        }
                    }
                } 
                else {
                    if (s1 === source.data['weeks'][i]) {
                        if (!(s2 === undefined || s2.length == 0)) {
                            if(s2.indexOf(source.data['player_name'][i]) >= 0) { 
                                if(sc.data['index'].indexOf(source.data['index'][i]) < 0) {
                                    for (const col in source.data) {
                                        sc.data[col].push(source.data[col][i]);                
                                    }
                                }
                                if(sc2.get_length() < 32 && sc2.data['index'].indexOf(source.data['index'][i]) < 0) {
                                    for (const col in source.data) {
                                        sc2.data[col].push(source.data[col][i]);                
                                    }
                                }
                            }
                            else {
                                if(sc.data['index'].indexOf(source.data['index'][i]) >= 0) {
                                    var q = sc.data['index'].indexOf(source.data['index'][i]);
                                    for (const col in source.data) {
                                        sc.data[col].splice(q, 1);                
                                    }
                                    
                                }
                                if(sc2.data['index'].indexOf(source.data['index'][i]) >= 0) {
                                    var q = sc2.data['index'].indexOf(source.data['index'][i]);
                                    for (const col in source.data) {
                                        sc2.data[col].splice(q, 1);                
                                    }
                                    
                                }
                            }
                        }
                        else {
                            if(sc.data['index'].indexOf(source.data['index'][i]) < 0) {
                                pl.push(source.data['player_name'][i]);
                                for (const col in source.data) {
                                    sc.data[col].push(source.data[col][i]);                
                                }
                            }
                            if(sc2.get_length() < 32 && sc2.data['index'].indexOf(source.data['index'][i]) < 0) {
                                pl2.push(source.data['player_name'][i]);
                                for (const col in source.data) {
                                    sc2.data[col].push(source.data[col][i]);                
                                }
                            }
                        }                                    
                    }
                }
            }
            var pl1 = [...new Set(pl)]; 
            var pll2 = [...new Set(pl2)];
            if (!(s2 === undefined || s2.length == 0)) {
                p1.x_range.factors = s2;
                p2.x_range.factors = s2;
            }
            else {
                p1.x_range.factors = pl1;
                p2.x_range.factors = pll2;
            }
            const inlo1 = Math.min(...sc.data['predicted_fantasy'])
            const inlo2 = Math.min(...sc.data['fantasy_points_ppr'])
            const inlo3 = Math.min(...sc2.data['predicted_fantasy'])
            lo = (lo > inlo1 && inlo1 < 0) ? inlo1 : 0;
            lo = (lo > inlo2 && inlo2 < 0) ? inlo2 : 0;
            lo2 = (lo2 > inlo3 && inlo3 < 0) ? inlo3 : 0;
            p1.y_range.start = lo;
            p2.y_range.start = lo2;
            p3.y_range.start = lo;
        }
        update();
        sc.change.emit();
        sc2.change.emit();
        
    """
    #Dropdowns lists
    sort_df1 = sorted(src['week'].unique().tolist())
    sort_df1 = ['Week ' + str(i) for i in sort_df1]
    sort_df1 = ['All'] + sort_df1
    sort_df2 = sorted(src['player_name'].astype(str).unique().tolist())
    #Selects players
    multi_select_player = MultiSelect(title="Players to plot:", height = 200, \
        value=[], \
        options=sort_df2)
    #Buttons and selects
    multi_select_player2 = MultiSelect(title="Selected Players:", height = 200, \
        value=[], \
        options=[])
    pret = PreText(text="", \
            width=200, styles={'color': 'red', 'font-size': '9pt'})
    but1 = Button(label="Add", button_type="success")
    but2 = Button(label="Remove", button_type="danger")
    but3 = Button(label="Visualize", button_type="primary")
    but1.js_on_event(ButtonClick, CustomJS(args=dict(t1 = multi_select_player, ms=multi_select_player2,tex = pret), code = """
                var a1 = t1.value;
                var pret = tex;
                var temp = [];
                for (const a of ms.options) {
                    temp.push(a);       
                }
                if (a1 == "undefined" || a1.length == 0) {
                    pret.text = "Select player name to add";
                }
                else {
                    pret.text = "";
                    for(const a of a1) {
                        if (temp.indexOf(a) < 0) {
                            temp.push(a);
                        }
                    }
                }
                ms.options = temp;
            """
        ))
    but2.js_on_event(ButtonClick, CustomJS(args=dict(ms=multi_select_player2,tex = pret), code = """
                var b1 = ms.value;
                var pret = tex;
                var temp = [];
                for (const a of ms.options) {
                    temp.push(a);       
                }
                if (b1 == "undefined" || b1.length == 0) {
                    pret.text = "Select player name to remove";
                }
                else {
                    pret.text = "";
                    for(const a of b1) {
                        if (temp.indexOf(a) >= 0) {
                            temp.splice(temp.indexOf(a), 1);
                        }
                    }
                }
                ms.options = temp;
                ms.value = [];
            """
        ))
    ti = TextInput(placeholder='Enter player_name'
                )
    ti.js_on_change('value', CustomJS(args=dict(ds=sort_df2, s=multi_select_player),
                                    code="s.options = ds.filter(i => i.toLowerCase().includes(cb_obj.value.toLowerCase()));"))
    select1 = Select(options= sort_df1, value = [], title = 'Week')
    p1 = figure(tools=TOOLS, toolbar_location="above", width=1200, height = 725,x_range = FactorRange(factors=sorted(list(set(stateChange.data['player_name'])))),title="NFL Fantasy Points Predicted for Players (Season 2024)")
    p1.toolbar.logo = "grey"
    p1.background_fill_color = "#d6cdcb"
    p1.xaxis.axis_label = "Players"
    p1.yaxis.axis_label = "Prediction points"
    p1.grid.grid_line_color = "white"
    p1.xgrid.grid_line_dash = 'dashed'
    p1.xaxis.major_label_orientation = math.pi/4
    p1.ygrid.grid_line_dash = 'dotted'
    p1.hover.tooltips = [
        ("Player name", "@player_name"),
        ("Predicted Point", "@predicted_fantasy{1.1}"),
        ("Fantasy Point:", "@fantasy_points_ppr{1.1}"),
        ("Positions", "@position"),
        ("Passing Yards", "@passing_yards"),
        ("Passing TD", "@passing_tds"),
        ("Rushing Yards", "@rushing_yards"),
        ("Rushing TD", "@rushing_tds"),
        ("Receptions", "@receptions"),
        ("Interceptions", "@interceptions"),
    ]
    scatterplot = p1.scatter("player_name", "predicted_fantasy", size=10, source=stateChange, legend_label = 'predicted_fantasy',
            color='colors1', line_color="black", alpha=0.6)
    scatterplot2 = p1.scatter("player_name", "fantasy_points_ppr", size=10, source=stateChange, legend_label = 'Fantasy Points',
            color='colors2', line_color="black", alpha=0.6)

    p1.legend.location = "top_left"
    p1.legend.click_policy="hide"
    p1.legend.label_text_font = "times"
    p1.legend.label_text_font_style = "italic"
    p1.legend.label_text_color = "navy"


    p1.legend.background_fill_color = "#d6cdcb"
    p1.legend.background_fill_alpha = 0.2

    week_l = sorted(src['week'].unique().tolist())
    week_l = ['Week ' + str(i) for i in week_l]
    p3 = figure(tools=TOOLS, toolbar_location="above", width=1200, height = 725, x_range = week_l, title="NFL Fantasy Points Predicted for Weeks (Season 2024)")
    p3.toolbar.logo = "grey"
    p3.background_fill_color = "#d6cdcb"
    p3.xaxis.axis_label = "Weeks"
    p3.yaxis.axis_label = "Prediction points"
    p3.grid.grid_line_color = "white"
    p3.xgrid.grid_line_dash = 'dashed'
    p3.xaxis.major_label_orientation = math.pi/4
    p3.ygrid.grid_line_dash = 'dotted'
    p3.hover.tooltips = [
        ("Player name", "@player_name"),
        ("Predicted Point", "@predicted_fantasy{1.1}"),
        ("Fantasy Point:", "@fantasy_points_ppr{1.1}"),
        ("Positions", "@position"),
        ("Passing Yards", "@passing_yards"),
        ("Passing TD", "@passing_tds"),
        ("Rushing Yards", "@rushing_yards"),
        ("Rushing TD", "@rushing_tds"),
        ("Receptions", "@receptions"),
        ("Interceptions", "@interceptions"),
    ]
    spl = p3.scatter("weeks", "predicted_fantasy", size=10, source=stateChange,legend_label = 'predicted_fantasy',
            color='colors1', line_color="black", alpha=0.6)
    spl2 = p3.scatter("weeks", "fantasy_points_ppr", size=10, source=stateChange,legend_label = 'Fantasy Points',
            color='colors2', line_color="black", alpha=0.6)

    p3.legend.location = "top_left"
    p3.legend.click_policy="hide"
    p3.legend.label_text_font = "times"
    p3.legend.label_text_font_style = "italic"
    p3.legend.label_text_color = "navy"

    p3.legend.background_fill_color = "#d6cdcb"
    p3.legend.background_fill_alpha = 0.2
    player_list = list(sc2.data['player_name'])
    p2 = figure(x_range = FactorRange(factors=list(set(sorted(player_list, key=lambda x: sc2.data['predicted_fantasy'][player_list.index(x)], reverse=True)))), tools=TOOLS, toolbar_location="above", x_axis_label = 'Player Name', y_axis_label = 'Fantasy Points',  width=1200, height = 725, title="NFL Fantasy Points Predicted Fantasy Bar Plot on players (Season 2024)")
    p2.grid.grid_line_color = "white"
    p2.toolbar.logo = "grey"
    p2.background_fill_color = "#d6cdcb"
    p2.xgrid.grid_line_dash = 'dashed'
    p2.ygrid.grid_line_dash = 'dotted'

    p2.xaxis.major_label_orientation = math.pi/4

    p2.hover.tooltips = [
        ("Player name", "@player_name"),
        ("Predicted Point", "@predicted_fantasy{1.1}"),
        ("Fantasy Point:", "@fantasy_points_ppr{1.1}"),
        ("Positions", "@position"),
        ("Passing Yards", "@passing_yards"),
        ("Passing TD", "@passing_tds"),
        ("Rushing Yards", "@rushing_yards"),
        ("Rushing TD", "@rushing_tds"),
        ("Receptions", "@receptions"),
        ("Interceptions", "@interceptions"),
    ]

    # plot bar chart glyph
    p2.vbar(x = 'player_name', top = 'predicted_fantasy', width=0.3, color='colors3', source=sc2, alpha = 1.2)
    callback = CustomJS(args=dict(source=source, sc=stateChange, sel1 = select1, sel2 = multi_select_player2, p1 = p1, p2 = p2,p3=p3, sc2=sc2,pr=pret), code = code_js)
    but3.js_on_event(ButtonClick, callback)
    #Group in tab panel
    tabs = Tabs(tabs=[
        TabPanel(child=p1, title="Scatter Player Name", tooltip=Tooltip(content="Scatter player name Plot Visualize fantasy predicted points", position="bottom_center")),
        TabPanel(child=p3, title="Scatter Week", tooltip=Tooltip(content="Scatter week Plot Visualize fantasy predicted points", position="bottom_center")),
        TabPanel(child=p2, title="Vertical Bar", tooltip=Tooltip(content="Vbar Plot Visualize fantasy predicted points", position="bottom_center")),
    ])
    columns = [
                TableColumn(field="player_name", title="Players_name"),
                TableColumn(field="week", title="Week"),
                TableColumn(field="season", title="Seasons"),
                TableColumn(field="fantasy_points_ppr", title="Fantasy points"),
                TableColumn(field="predicted_fantasy", title="Prediction Points"),
            ]

    # data table for scatter
    dtab = DataTable(source=stateChange, columns=columns, width=600, height=200)
    # data table for bar
    columns = [
                TableColumn(field="player_name", title="Players_name"),
                TableColumn(field="week", title="Week"),
                TableColumn(field="season", title="Seasons"),
                TableColumn(field="fantasy_points_ppr", title="Fantasy points"),
                TableColumn(field="predicted_fantasy", title="Prediction Points"),
            ]

    # data table
    dtab2 = DataTable(source=sc2, columns=columns, width=600, height=200)
    d1 = Button(label="Download Data for Scatter", button_type="primary")
    d2 = Button(label="Download Data for Bar Plot", button_type="primary")

    downloadCode = """
        function convertToCSV(objArray) {
            console.log(typeof objArray);
            var array = objArray;
            var str = '';
            console.log(array.get_length());
            for(const col in array.data) {
                if (col === 'colors1') break;
                str = (col === 'weeks') ? str + col : str + col + ',';
            }
            str += '\\r\\n';         
            for (var i = 0; i < array.get_length(); i++) {
                var line = '';
                for (var index in array.data) {
                    if (index === 'colors1') break; 
                    if (line != '') line += ','
                    var d = array.data[index][i];
                    d = String(d);
                    d = d.split(",").join("-"); 
                    line += d;
                }
                str += line + '\\r\\n';
            }
            return str;
        }
        var data = source;
        var csv = convertToCSV(data);
        var blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        var link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = (togg === 'scatter') ? 'ScatterData.csv' : 'BarData.csv';
        link.click();
    """
    d1.js_on_click(CustomJS(args=dict(source=stateChange,togg='scatter'), code=downloadCode))
    d2.js_on_click(CustomJS(args=dict(source=sc2,togg='vbar'), code=downloadCode))
    div1 = Div(text="""<img src='logo.png'>""")

    control = column(tabs,row(column(div2,dtab, d1), column(div3,dtab2, d2)))
    layout=row(column(div1,select1,ti,multi_select_player, pret,row(but1,but2),multi_select_player2, but3), row(control))
    save(layout)  # save the plot
    show(layout)


visualize(datam = './fantasy_prediction_data.csv', viz = 'index.html')
