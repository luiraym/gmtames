function varargout = ad_gui(varargin)

% ad_gui opens a GUI figure for calculating Applicability Domain;
% in order to open the graphical interface, just type on the matlab command line:
%
% ad_gui
%
% there are no inputs, data can be loaded and saved directly from the graphical interface
% 
% Note that a detailed HTML help is provided with the toolbox.
% See the HTML HELP files (help.htm) for futher details and examples
%
% The toolbox is freeware and may be used (but not modified) if proper reference is given to the authors. 
% Preferably refer to the following papers:
%
% F. Sahigara, D. Ballabio, R. Todeschini, V. Consonni
% Assessing the validity of QSARs for ready biodegradability of chemicals: an Applicability Domain perspective
% Current Computer-Aided Drug Design (2014), 10, 137-147
% 
% F. Sahigara, K. Mansouri, D. Ballabio, A. Mauri, V. Consonni, R. Todeschini
% Comparison of different approaches to define the Applicability Domain of QSAR models
% Molecules (2012), 17, 4791-4810
% 
% Applicabilit domain toolbox for MATLAB
% version 1.0 - january 2014
% Milano Chemometrics and QSAR Research Group
% http://michem.disat.unimib.it/chm/

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ad_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @ad_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin & isstr(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before ad_gui is made visible.
function ad_gui_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;
movegui('center');

% initialize handles
handles = init_handles(handles);

% enable/disable buttons and menu
handles = enable_disable(handles);

% updtae list boxes
update_listbox_data(handles)
update_listbox_ad(handles)

% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = ad_gui_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;

% --------------------------------------------------------------------
function m_file_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function m_file_load_train_Callback(hObject, eventdata, handles)
% ask for overwriting
if handles.present.train == 1
    q = questdlg('Training set is alreday loaded. Do you wish to overwrite it?','loading training','yes','no','yes');
else
    q = 'yes';
end
if strcmp(q,'yes')
    res = visualize_load(1,[0 0]);
    if isnan(res.loaded_file)
        if handles.present.train  == 0
            handles.present.train  = 0;
        else
            handles.present.train  = 1;
        end
    elseif res.from_file == 1
        handles = init_handles(handles);
        handles.present.train  = 1;
        tmp_data = load(res.path);
        handles.data.Xtrain = getfield(tmp_data,res.name);
        handles.data.name_train = res.name;
    else
        handles = init_handles(handles);
        handles.present.train  = 1;
        handles.data.Xtrain = evalin('base',res.name);
        handles.data.name_train = res.name;
    end
    handles = enable_disable(handles);
    update_listbox_data(handles)
    update_listbox_ad(handles)
    guidata(hObject,handles)
end

% --------------------------------------------------------------------
function m_file_load_test_Callback(hObject, eventdata, handles)
% ask for overwriting
if handles.present.test == 1
    q = questdlg('Test set is alreday loaded. Do you wish to overwrite it?','loading test','yes','no','yes');
else
    q = 'yes';
end
if strcmp(q,'yes')
    res = visualize_load(2,size(handles.data.Xtrain));
    if isnan(res.loaded_file)
        if handles.present.test  == 0
            handles.present.test  = 0;
        else
            handles.present.test  = 1;
        end
    elseif res.from_file == 1
        handles = reset_test(handles);
        handles.present.test  = 1;
        tmp_data = load(res.path);
        handles.data.Xtest = getfield(tmp_data,res.name);
        handles.data.name_test = res.name;
    else
        handles = reset_test(handles);
        handles.present.test  = 1;
        handles.data.Xtest = evalin('base',res.name);
        handles.data.name_test = res.name;
    end
    handles = enable_disable(handles);
    update_listbox_data(handles)
    update_listbox_ad(handles)
    guidata(hObject,handles)
end

% --------------------------------------------------------------------
function m_file_save_ad_Callback(hObject, eventdata, handles)
visualize_export(handles.data.ad,'ad')

% --------------------------------------------------------------------
function m_file_export_test_Callback(hObject, eventdata, handles)
visualize_export(handles.data.ad.resume_table,'export')

% --------------------------------------------------------------------
function m_file_clear_data_Callback(hObject, eventdata, handles)
handles = init_handles(handles);
handles = enable_disable(handles);
update_listbox_data(handles)
update_listbox_ad(handles)
guidata(hObject,handles)

% --------------------------------------------------------------------
function m_file_exit_Callback(hObject, eventdata, handles)
close

% --------------------------------------------------------------------
function m_calculate_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function m_calculate_ad_Callback(hObject, eventdata, handles)
handles = do_ad(handles);
guidata(hObject,handles)

% --------------------------------------------------------------------
function m_results_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function m_results_view_test_Callback(hObject, eventdata, handles)
assignin('base','tmp_view',handles.data.ad.resume_table);
openvar('tmp_view');

% --------------------------------------------------------------------
function m_results_consensus_Callback(hObject, eventdata, handles)
disp_consensus(handles)

% --------------------------------------------------------------------
function m_results_william_Callback(hObject, eventdata, handles)
handles = disp_william(handles);
guidata(hObject,handles)

% --------------------------------------------------------------------
function m_results_k_Callback(hObject, eventdata, handles)
disp_k(handles)

% --------------------------------------------------------------------
function m_help_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function m_help_html_Callback(hObject, eventdata, handles)
h1 = ['A complete HTML guide on how to use' sprintf('\n') 'the Applicability Domain toolbox is provided.'];
hr = sprintf('\n');
h3 = ['Look for the help.htm file in the toolbox folder' sprintf('\n') 'and open it in your favourite browser!'];
helpdlg([h1 hr h3],'HTML help')

% --------------------------------------------------------------------
function m_help_cite_Callback(hObject, eventdata, handles)
h1 = ['The toolbox is freeware and may be used (but not modified) if proper reference is given to the authors. Preferably refer to the following papers:'];
hr = sprintf('\n');
h3 = ['F. Sahigara, D. Ballabio, R. Todeschini, V. Consonni, Assessing the validity of QSARs for ready biodegradability of chemicals: an Applicability Domain perspective, Accepted for publication in Current Computer-Aided Drug Design'];
h4 = ['F. Sahigara, K. Mansouri, D. Ballabio, A. Mauri, V. Consonni, R. Todeschini, Comparison of different approaches to define the Applicability Domain of QSAR models, Molecules (2012), 17, 4791-4810 '];
helpdlg([h1 hr hr h3 hr hr h4 hr hr],'HTML help')

% --------------------------------------------------------------------
function m_about_Callback(hObject, eventdata, handles)
h1 = 'Applicability domain toolbox for MATLAB ver. 1.0';
hr = sprintf('\n');
h2 = 'Milano Chemometrics and QSAR Research Group ';
h3 = 'University of Milano-Bicocca, Italy';
h4 = 'http://michem.disat.unimib.it/chm';
helpdlg([h1 hr h2 hr h3 hr h4],'HTML help')

% --------------------------------------------------------------------
function listbox_data_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% --------------------------------------------------------------------
function listbox_data_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function listbox_ad_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% --------------------------------------------------------------------
function listbox_ad_Callback(hObject, eventdata, handles)

% ------------------------------------------------------------------------
function update_listbox_data(handles)
if handles.present.train == 0
    str{1} = 'data not loaded';
else
    str{1} = ['training set: loaded'];
    str{2} = ['name: ' handles.data.name_train];
    str{3} = ['samples: ' num2str(size(handles.data.Xtrain,1))];
    str{4} = ['variables: ' num2str(size(handles.data.Xtrain,2))];
    if handles.present.test == 0
        str{5} = ['test set: not loaded'];
    else
        str{5} = ['test set: loaded'];
        str{6} = ['name: ' handles.data.name_test];
        str{7} = ['samples: ' num2str(size(handles.data.Xtest,1))];
        str{8} = ['variables: ' num2str(size(handles.data.Xtest,2))];
    end
end
set(handles.listbox_data,'String',str);

% ------------------------------------------------------------------------
function update_listbox_ad(handles)
whattodo = handles.data.whattodo;
if handles.present.ad == 0
    str= 'AD not calculated';
else
    cnt_row = 1;
    str{cnt_row} = ['AD: calculated'];
    cnt_row = cnt_row + 1;
    str{cnt_row} = ['Test outsude AD:'];
    if whattodo.bounding_box
        cnt_row = cnt_row + 1;
        str{cnt_row} = ['Bounding box: ',num2str(length(find(handles.data.ad.bounding_box.inad==0)))];
    end
    if whattodo.bounding_box_pca
        cnt_row = cnt_row + 1;
        str{cnt_row} = ['Bounding box PCA: ',num2str(length(find(handles.data.ad.bounding_box_pca.inad==0)))];
    end
    if whattodo.convex_hull
        cnt_row = cnt_row + 1;
        str{cnt_row} = ['Convex hull: ',num2str(length(find(handles.data.ad.convex_hull.inad==0)))];
    end
    if whattodo.leverage
        cnt_row = cnt_row + 1;
        str{cnt_row} = ['Leverage: ',num2str(length(find(handles.data.ad.leverage.inad==0)))];
    end
    if whattodo.dist_centroid
        cnt_row = cnt_row + 1;
        str{cnt_row} = ['Distance from centroid: ',num2str(length(find(handles.data.ad.dist_centroid.inad==0)))];
    end
    if whattodo.dist_knn_fix
        cnt_row = cnt_row + 1;
        str{cnt_row} = ['kNN - fixed k: ',num2str(length(find(handles.data.ad.dist_knn_fix.inad==0)))];
    end
    if whattodo.dist_knn_var
        cnt_row = cnt_row + 1;
        str{cnt_row} = ['kNN - variable k: ',num2str(length(find(handles.data.ad.dist_knn_var.inad==0)))];
    end
    if whattodo.pot_fun
        cnt_row = cnt_row + 1;
        str{cnt_row} = ['Potential functions: ',num2str(length(find(handles.data.ad.pot_fun.inad==0)))];
    end
end
set(handles.listbox_ad,'String',str);

% ------------------------------------------------------------------------
function handles = init_handles(handles)
handles.present.train = 0;
handles.present.test = 0;
handles.present.ad = 0;
handles.data.name_train = [];
handles.data.name_test = [];
handles.data.name_ad = [];
handles.data.Xtrain = [];
handles.data.Xtest = [];
handles.data.ad = [];
handles.data.whattodo = [];
handles.data.y_train_exp = [];
handles.data.y_train_calc = [];
handles.data.y_test_exp = [];
handles.data.y_test_calc = [];
handles.data.y_train_exp_name = '';
handles.data.y_train_calc_name = '';
handles.data.y_test_exp_name = '';
handles.data.y_test_calc_name = '';

% ------------------------------------------------------------------------
function handles = reset_test(handles);
handles.present.test = 0;
handles.present.ad = 0;
handles.data.name_test = [];
handles.data.Xtest = [];
handles.data.name_ad = [];
handles.data.ad = [];
handles.data.whattodo = [];
handles.data.y_test_exp = [];
handles.data.y_test_calc = [];
handles.data.y_test_exp_name = '';
handles.data.y_test_calc_name = '';

% ------------------------------------------------------------------------
function handles = enable_disable(handles)
if handles.present.train == 0
    set(handles.m_file_load_test,'Enable','off');    
    set(handles.m_file_clear_data,'Enable','off');
else
    set(handles.m_file_load_test,'Enable','on');
    set(handles.m_file_clear_data,'Enable','on');
end

if handles.present.test == 0
    set(handles.m_calculate_ad,'Enable','off');
else
    set(handles.m_calculate_ad,'Enable','on');
end

if handles.present.ad == 0
    set(handles.m_file_save_ad,'Enable','off');
    set(handles.m_file_export_test,'Enable','off');    
    set(handles.m_results_view_test,'Enable','off');
    set(handles.m_results_consensus,'Enable','off');
    set(handles.m_results_william,'Enable','off');
    set(handles.m_results_k,'Enable','off');
else
    set(handles.m_file_save_ad,'Enable','on');
    set(handles.m_file_export_test,'Enable','on');    
    set(handles.m_results_view_test,'Enable','on');
    set(handles.m_results_consensus,'Enable','on');
    if handles.data.whattodo.leverage
        set(handles.m_results_william,'Enable','on');
    else
        set(handles.m_results_william,'Enable','off');
    end
    if handles.data.whattodo.dist_knn_var
        if isstr(handles.data.ad.options.knnvar_k_opt)
            set(handles.m_results_k,'Enable','on');
        else
            set(handles.m_results_k,'Enable','off');
        end
    else
        set(handles.m_results_k,'Enable','off');
    end
end

% ------------------------------------------------------------------------
function handles = do_ad(handles)
% open do settings
maxcomp = min([size(handles.data.Xtrain,2) size(handles.data.Xtrain,1)]);
if maxcomp > 20; maxcomp = 20; end
[options,whattodo,doad] = visualize_settings(maxcomp,size(handles.data.Xtrain,2),handles.present.ad,handles.data.ad,handles.data.whattodo);
if doad
    % activate pointer
    set(handles.ad_gui,'Pointer','watch')
    % do ad calculation
    res_ad = ad_model(handles.data.Xtrain,handles.data.Xtest,options,whattodo);
    if isstruct(res_ad)
        % store results
        handles.present.ad = 1;
        handles.data.name_ad = 'ad';
        handles.data.ad = res_ad;
        handles.data.whattodo = whattodo;
    end
    % update model listbox
    set(handles.ad_gui,'Pointer','arrow')
    update_listbox_ad(handles)
    handles = enable_disable(handles);
else
    set(handles.ad_gui,'Pointer','arrow')
end

% ------------------------------------------------------------------------
function disp_consensus(handles)
% consensus plot
whattodo = handles.data.whattodo;
cnt_row = 0;
if whattodo.bounding_box
    cnt_row = cnt_row + 1;
    S(cnt_row,:) = handles.data.ad.bounding_box.inad';
end
if whattodo.bounding_box_pca
    cnt_row = cnt_row + 1;
    S(cnt_row,:) = handles.data.ad.bounding_box_pca.inad';
end
if whattodo.convex_hull
    cnt_row = cnt_row + 1;
    S(cnt_row,:) = handles.data.ad.convex_hull.inad';
end
if whattodo.leverage
    cnt_row = cnt_row + 1;
    S(cnt_row,:) = handles.data.ad.leverage.inad';
end
if whattodo.dist_centroid
    cnt_row = cnt_row + 1;
    S(cnt_row,:) = handles.data.ad.dist_centroid.inad';
end
if whattodo.dist_knn_fix
    cnt_row = cnt_row + 1;
    S(cnt_row,:) = handles.data.ad.dist_knn_fix.inad';
end
if whattodo.dist_knn_var
    cnt_row = cnt_row + 1;
    S(cnt_row,:) = handles.data.ad.dist_knn_var.inad';
end
if whattodo.pot_fun
    cnt_row = cnt_row + 1;
    S(cnt_row,:) = handles.data.ad.pot_fun.inad';
end
S = abs(1 - S);
freq = sum(S);
[freq,samples] = sort(-freq);
freq = - freq;
h = find(freq > 0);
freq = freq(h);
samples = samples(h);
for k=1:length(samples); labels{k}= num2str(samples(k));end
figure
bar(freq)
title ('Consensus outliers')
xlabel('test samples')
set(gca,'XTick',[1:length(freq)])
set(gca,'XTickLabel',labels)
set(gcf,'color','white')

% ------------------------------------------------------------------------
function handles = disp_william(handles)
% open do settings
[data,dowilliam,addlabels] = visualize_william(handles.data);
if dowilliam
    % william plot
    [r_std_train,r_std_test] = std_residuals(data.y_train_exp,data.y_train_calc,data.y_test_exp,data.y_test_calc,size(handles.data.Xtrain,2),handles.data.ad.leverage.h_train,handles.data.ad.leverage.h_test);
    figure
    set(gcf,'color','white'); box on;
    hold on
    plot(handles.data.ad.leverage.h_train,r_std_train,'o','MarkerEdgeColor','k','MarkerSize',5,'MarkerFaceColor','k')
    plot(handles.data.ad.leverage.h_test,r_std_test,'o','MarkerEdgeColor','k','MarkerSize',5,'MarkerFaceColor','r')
    % set max and min for axis
    range_plot_x = [handles.data.ad.leverage.h_train;handles.data.ad.leverage.h_test];
    range_plot_y = [r_std_train r_std_test];
    range_x = max(range_plot_x) - min(range_plot_x); add_space_x = range_x/20;      
    x_lim = [min(range_plot_x)-add_space_x max(range_plot_x)+add_space_x];
    range_y = max(range_plot_y) - min(range_plot_y); add_space_y = range_y/20;      
    y_lim = [min(range_plot_y)-add_space_y max(range_plot_y)+add_space_y];    
    this = find_max_axis(x_lim(2),handles.data.ad.leverage.thr);
    x_lim = [0 this];
    if y_lim(1) > -2; y_lim(1) = -2.1; end
    if y_lim(2) < 2; y_lim(2) = 2.1; end    
    line([handles.data.ad.leverage.thr handles.data.ad.leverage.thr],y_lim,'Color','r','LineStyle',':')
    line(x_lim,[0 0],'Color','r','LineStyle',':')
    line(x_lim,[2 2],'Color','r','LineStyle',':')
    line(x_lim,[-2 -2],'Color','r','LineStyle',':')
    axis([x_lim(1) x_lim(2) y_lim(1) y_lim(2)])
    if addlabels
        range_span = (max(range_plot_x) - min(range_plot_x));
        plot_string_label(handles.data.ad.leverage.h_train,r_std_train,'k',range_span);
        plot_string_label(handles.data.ad.leverage.h_test,r_std_test,'r',range_span);
    end
    title('William plot - train samples in black, test samples in red')
    xlabel('Leverage')
    ylabel('Standardised residuals')
    hold off
    handles.data.y_train_exp = data.y_train_exp;
    handles.data.y_train_calc = data.y_train_calc;
    handles.data.y_test_exp = data.y_test_exp;
    handles.data.y_test_calc = data.y_test_calc;
    handles.data.y_train_exp_name = data.y_train_exp_name;
    handles.data.y_train_calc_name = data.y_train_calc_name;
    handles.data.y_test_exp_name = data.y_test_exp_name;
    handles.data.y_test_calc_name = data.y_test_calc_name;
end

% ------------------------------------------------------------------------
function [r_std_train,r_std_test] = std_residuals(ytrain,ytrain_calc,ytest,ytest_calc,nvar,htrain,htest)
nobj_train = length(ytrain);
nobj_test = length(ytest);
r_train = ytrain - ytrain_calc;
r_test = ytest - ytest_calc;
RSS = sum(r_train.^2);
s = sqrt(RSS/(nobj_train - nvar));
for k=1:nobj_train; r_std_train(k) = r_train(k)/(s*(1 - htrain(k))^0.5); end
for k=1:nobj_test; r_std_test(k) = r_test(k)/(s*(1 - htest(k))^0.5); end

% ------------------------------------------------------------------------
function plot_string_label(x,y,col,range_span)
add_span = range_span/100;
for j=1:length(x); text(x(j)+add_span,y(j),num2str(j),'Color',col); end;

% ------------------------------------------------------------------------
function this = find_max_axis(x1,x2);
m = max([x1 x2]);
this = m + m/20;

% ------------------------------------------------------------------------
function disp_k(handles)
% k plot
k_optimisation = handles.data.ad.dist_knn_var.k_optimisation;
figure
boxplot(k_optimisation);
set(gcf,'color','white'); box on;
hold on
plot(mean(k_optimisation),'-*');
xlabel(' k values');
ylabel(' Distribution of samples inside the AD (%)');
title(' Boxplot for the samples inside AD in k optimization');
m = min(min(k_optimisation));
M = max(max(k_optimisation));
R = (M - m)/20;
m = m - R;
M = M + R;
if M > 100; M = 100; end
axis([0 size(k_optimisation,2)+0.5 m M])
hold off
