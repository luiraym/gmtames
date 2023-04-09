function varargout = visualize_settings(varargin)

% visualize_settings opens a graphical interface for prepearing
% settings of AD
%
% The main function for the AD toolbox is ad_gui, which opens a GUI figure for calculating Applicability Domain;
% in order to open the graphical interface, just type on the matlab command line: ad_gui
% 
% Note that a detailed HTML help is provided with the toolbox.
% See the HTML HELP files (help.htm) for futher details and examples
%
% Applicabilit domain toolbox for MATLAB
% version 1.0 - january 2014
% Milano Chemometrics and QSAR Research Group
% http://michem.disat.unimib.it/chm/

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @visualize_settings_OpeningFcn, ...
                   'gui_OutputFcn',  @visualize_settings_OutputFcn, ...
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


% --- Executes just before visualize_settings is made visible.
function visualize_settings_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
movegui(handles.visualize_settings,'center')
handles.maxcomp = varargin{1};
handles.numvar = varargin{2};
handles.ad_is_present = varargin{3};
handles.ad_loaded = varargin{4};
handles.ad_loaded.whattodo = varargin{5};

% set scaling combo
str_disp={};
str_disp{1} = 'none';
str_disp{2} = 'mean centering';
str_disp{3} = 'autoscaling';
str_disp{4} = 'range scaling';
set(handles.pop_menu_scaling,'String',str_disp);

% set distance combo
str_disp={};
str_disp{1} = 'euclidean';
str_disp{2} = 'manhattan';
str_disp{3} = 'mahalanobis';
set(handles.pop_menu_distance,'String',str_disp);

% set percentile combo
str_disp={};
str_disp{1} = '99';
str_disp{2} = '95';
str_disp{3} = '90';
str_disp{4} = '85';
str_disp{5} = '80';
set(handles.pop_menu_pct,'String',str_disp);

% set k combo
str_disp={};
max_k = 25;
for j=1:max_k
    str_disp{j} = num2str(j);
end
set(handles.pop_menu_k,'String',str_disp);

% set k optimal combo
str_disp={};
str_disp{1} = 'automatic';
str_disp{2} = 'user';
for j=1:25
    str_disp{j+2} = num2str(j);
end
set(handles.pop_menu_kopt,'String',str_disp);

% set potential functions combo
str_disp={};
str_disp{1} = 'gaussian';
str_disp{2} = 'triangular';
set(handles.pop_menu_pot_fun,'String',str_disp);

% set pop_menu_smoothing combo
str_disp={};
str_disp{1} = 'automatic';
cnt = 1;
for j=0.2:0.1:1.4
    cnt = cnt + 1;
    str_disp{cnt} = num2str(j);
end
set(handles.pop_menu_smoothing,'String',str_disp);

% set validation combo
str_disp={};
for j=1:7
    p = 5 + j*5;
    str_disp{j} = num2str(p);
end
set(handles.pop_menu_validation,'String',str_disp);

% set iteration combo
str_disp={};
str_disp{1} = '10';
str_disp{2} = '50';
str_disp{3} = '100';
str_disp{4} = '500';
str_disp{5} = '1000';
set(handles.pop_menu_iter,'String',str_disp);

% set maximum k combo
str_disp={};
max_k = 25;
for j=1:max_k
    str_disp{j} = num2str(j);
end
set(handles.pop_menu_maxk,'String',str_disp);

% initialize values
handles.doad = 0;
if handles.ad_is_present == 1
    handles = init_handles_ad_loaded(handles);
else
    set(handles.pop_menu_distance,'Value',1);
    set(handles.pop_menu_scaling,'Value',3);
    set(handles.chk_bounding,'Value',1);
    set(handles.chk_bounding_pca,'Value',1);
    set(handles.chk_convex,'Value',1);
    set(handles.chk_leverage,'Value',1);
    set(handles.edit_txt_leverage,'String',num2str(2.5));
    set(handles.chk_distance_pct,'Value',1);
    set(handles.pop_menu_pct,'Value',2);
    set(handles.chk_distance_knn,'Value',1);
    set(handles.pop_menu_k,'Value',5);
    set(handles.chk_knn_var,'Value',1);
    set(handles.pop_menu_validation,'Value',3);
    set(handles.pop_menu_iter,'Value',5);
    set(handles.pop_menu_maxk,'Value',max_k);
    set(handles.pop_menu_kopt,'Value',1);
    set(handles.chk_pot_fun,'Value',1);
    set(handles.pop_menu_pot_fun,'Value',1);
    set(handles.pop_menu_smoothing,'Value',1);
end

handles = enable_disable(handles);

% prepare advanced show
handles.show_advanced = 0;
update_advanced_form(handles)

guidata(hObject, handles);
uiwait(handles.visualize_settings);

% --- Outputs from this function are returned to the command line.
function varargout = visualize_settings_OutputFcn(hObject, eventdata, handles)
len = length(handles);
if len > 0
    varargout{1} = handles.ad_options;
    varargout{2} = handles.whattodo;
    varargout{3} = handles.doad;
    delete(handles.visualize_settings)
else
    handles.ad_options = NaN;
    handles.whattodo = NaN;
    handles.doad = 0;
    varargout{1} = handles.ad_options;
    varargout{2} = handles.whattodo;
    varargout{3} = handles.doad;
end

% --- Executes on button press in button_calculate_ad.
function button_calculate_ad_Callback(hObject, eventdata, handles)
errortype = 'none';
errortype = check_errors(handles);
if strcmp(errortype,'none')
    [handles.ad_options,handles.whattodo] = make_options(handles);
    handles.doad = 1;
    guidata(hObject,handles)
    uiresume(handles.visualize_settings)
else
    errordlg(errortype,'loading error') 
end

% --- Executes on button press in button_cancel.
function button_cancel_Callback(hObject, eventdata, handles)
handles.ad_options = NaN;
handles.whattodo = NaN;
handles.doad = 0;
guidata(hObject,handles)
uiresume(handles.visualize_settings)

% --- Executes on button press in button_advanced.
function button_advanced_Callback(hObject, eventdata, handles)
handles.show_advanced = abs(handles.show_advanced - 1);
update_advanced_form(handles)
guidata(hObject, handles);

% ----------------------------------------------------------------
function pop_menu_numcomp_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% ----------------------------------------------------------------
function pop_menu_numcomp_Callback(hObject, eventdata, handles)

% ----------------------------------------------------------------
function pop_menu_scaling_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% ----------------------------------------------------------------
function pop_menu_scaling_Callback(hObject, eventdata, handles)

% ----------------------------------------------------------------
function pop_menu_distance_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% ----------------------------------------------------------------
function pop_menu_distance_Callback(hObject, eventdata, handles)

% ----------------------------------------------------------------
function chk_bounding_Callback(hObject, eventdata, handles)
handles = enable_disable(handles);
guidata(hObject,handles)

% ----------------------------------------------------------------
function chk_bounding_pca_Callback(hObject, eventdata, handles)
handles = enable_disable(handles);
guidata(hObject,handles)

% ----------------------------------------------------------------
function chk_convex_Callback(hObject, eventdata, handles)
handles = enable_disable(handles);
guidata(hObject,handles)

% ----------------------------------------------------------------
function chk_leverage_Callback(hObject, eventdata, handles)
handles = enable_disable(handles);
guidata(hObject,handles)

% ----------------------------------------------------------------
function chk_distance_pct_Callback(hObject, eventdata, handles)
handles = enable_disable(handles);
guidata(hObject,handles)

% ----------------------------------------------------------------
function chk_distance_knn_Callback(hObject, eventdata, handles)
handles = enable_disable(handles);
guidata(hObject,handles)

% ----------------------------------------------------------------
function chk_knn_var_Callback(hObject, eventdata, handles)
handles = enable_disable(handles);
guidata(hObject,handles)

% ----------------------------------------------------------------
function chk_pot_fun_Callback(hObject, eventdata, handles)
handles = enable_disable(handles);
guidata(hObject,handles)

% ----------------------------------------------------------------
function pop_menu_pct_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% ----------------------------------------------------------------
function pop_menu_pct_Callback(hObject, eventdata, handles)

% ----------------------------------------------------------------
function pop_menu_k_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% ----------------------------------------------------------------
function pop_menu_k_Callback(hObject, eventdata, handles)

% ----------------------------------------------------------------
function pop_menu_kopt_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% ----------------------------------------------------------------
function pop_menu_kopt_Callback(hObject, eventdata, handles)

% ----------------------------------------------------------------
function pop_menu_pot_fun_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% ----------------------------------------------------------------
function pop_menu_pot_fun_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function pop_menu_validation_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% --- Executes on selection change in pop_menu_validation.
function pop_menu_validation_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function pop_menu_iter_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% --- Executes on selection change in pop_menu_iter.
function pop_menu_iter_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function pop_menu_smoothing_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


% --- Executes on selection change in pop_menu_smoothing.
function pop_menu_smoothing_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function pop_menu_maxk_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% --- Executes on selection change in pop_menu_maxk.
function pop_menu_maxk_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function edit_txt_leverage_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% --- Executes during object creation, after setting all properties.
function edit_txt_leverage_Callback(hObject, eventdata, handles)

% ----------------------------------------------------------------
function handles = enable_disable(handles)

if get(handles.chk_leverage,'Value')
    set(handles.edit_txt_leverage,'Enable','on');
else
    set(handles.edit_txt_leverage,'Enable','off');
end

if get(handles.chk_distance_pct,'Value')
    set(handles.pop_menu_pct,'Enable','on');
else
    set(handles.pop_menu_pct,'Enable','off');
end

if get(handles.chk_distance_knn,'Value')
    set(handles.pop_menu_k,'Enable','on');
else
    set(handles.pop_menu_k,'Enable','off');
end

if get(handles.chk_knn_var,'Value')
    set(handles.pop_menu_kopt,'Enable','on');
    set(handles.pop_menu_maxk,'Enable','on');
    set(handles.pop_menu_iter,'Enable','on');
    set(handles.pop_menu_validation,'Enable','on');
else
    set(handles.pop_menu_kopt,'Enable','off');
    set(handles.pop_menu_maxk,'Enable','off');
    set(handles.pop_menu_iter,'Enable','off');
    set(handles.pop_menu_validation,'Enable','off');
end

if get(handles.chk_pot_fun,'Value')
    set(handles.pop_menu_pot_fun,'Enable','on');
    set(handles.pop_menu_smoothing,'Enable','on');
else
    set(handles.pop_menu_pot_fun,'Enable','off');
    set(handles.pop_menu_smoothing,'Enable','off');
end

if get(handles.chk_distance_pct,'Value')==0 & get(handles.chk_distance_knn,'Value')==0 & get(handles.chk_knn_var,'Value')==0
    set(handles.pop_menu_distance,'Enable','off');
else
    set(handles.pop_menu_distance,'Enable','on');
end

if get(handles.chk_bounding,'Value')==0 & get(handles.chk_bounding_pca,'Value')==0 & get(handles.chk_convex,'Value')==0 & get(handles.chk_leverage,'Value')==0 & get(handles.chk_distance_pct,'Value')==0 & get(handles.chk_distance_knn,'Value')==0 & get(handles.chk_knn_var,'Value')==0 & get(handles.chk_pot_fun,'Value')==0 
    set(handles.button_calculate_ad,'Enable','off');
else
    set(handles.button_calculate_ad,'Enable','on');
end

% ------------------------------------------------------------------
function update_advanced_form(handles)
width_full    = 113;
width_reduced = 68;
if handles.show_advanced == 1
    Pos = get(handles.visualize_settings,'Position');
    Pos(3) = width_full;
    set(handles.visualize_settings,'Position',Pos)
    set(handles.button_advanced,'String','hide advanced settings <<')
else
    Pos = get(handles.visualize_settings,'Position');
    Pos(3) = width_reduced;
    set(handles.visualize_settings,'Position',Pos)
    set(handles.button_advanced,'String','show advanced settings >>')
end

% -------------------------------------------------------------------------
function errortype = check_errors(handles)
errortype = 'none';
% advanced leverage threshold is a number
[i, status] = str2num(get(handles.edit_txt_leverage,'String'));
if status == 0 | isnan(i) | i < 0
    errortype = 'the leverage threshold must be a positive number';
    return
end
if get(handles.chk_convex,'Value') & handles.numvar > 10
    errortype = 'Convex hull can not be computed for data with more than 10 variables';
    return
end

% -------------------------------------------------------------------------
function [ad_options,whattodo] = make_options(handles)
% options
if get(handles.pop_menu_scaling,'Value') == 1
    ad_options.pret_type = 'none';
elseif get(handles.pop_menu_scaling,'Value') == 2
    ad_options.pret_type = 'cent';
elseif get(handles.pop_menu_scaling,'Value') == 3
    ad_options.pret_type = 'auto';
elseif get(handles.pop_menu_scaling,'Value') == 4
    ad_options.pret_type = 'rang';    
end
if get(handles.pop_menu_distance,'Value') == 1
    ad_options.distance = 'euclidean';
elseif get(handles.pop_menu_distance,'Value') == 2
    ad_options.distance = 'manhattan';
elseif get(handles.pop_menu_distance,'Value') == 3
    ad_options.distance = 'mahalanobis';
end
ad_options.lev_thr = str2num(get(handles.edit_txt_leverage,'String'));
ad_options.knnfix_k = get(handles.pop_menu_k,'Value');
if get(handles.pop_menu_pct,'Value') == 1
    ad_options.dist_pct = 99;
elseif get(handles.pop_menu_pct,'Value') == 2
    ad_options.dist_pct = 95;
elseif get(handles.pop_menu_pct,'Value') == 3
    ad_options.dist_pct = 90;
elseif get(handles.pop_menu_pct,'Value') == 4
    ad_options.dist_pct = 85;
elseif get(handles.pop_menu_pct,'Value') == 5
    ad_options.dist_pct = 80;
end
ad_options.knnvar_k_max = get(handles.pop_menu_maxk,'Value');
if get(handles.pop_menu_kopt,'Value') == 1
    ad_options.knnvar_k_opt = 'auto';
elseif get(handles.pop_menu_kopt,'Value') == 2
    ad_options.knnvar_k_opt = 'user';
else
    ad_options.knnvar_k_opt = get(handles.pop_menu_kopt,'Value') - 2;
end
if get(handles.pop_menu_iter,'Value') == 1
    ad_options.knnvar_iter = 10;
elseif get(handles.pop_menu_iter,'Value') == 2
    ad_options.knnvar_iter = 50;
elseif get(handles.pop_menu_iter,'Value') == 3
    ad_options.knnvar_iter = 100;
elseif get(handles.pop_menu_iter,'Value') == 4
    ad_options.knnvar_iter = 500;
elseif get(handles.pop_menu_iter,'Value') == 5
    ad_options.knnvar_iter = 1000;
end
p = get(handles.pop_menu_validation,'Value');
ad_options.knnvar_perc_test = 5 + p*5;
ad_options.pf_pct = ad_options.dist_pct;
if get(handles.pop_menu_pot_fun,'Value') == 1
    ad_options.pf_kernel = 'gaus';
elseif get(handles.pop_menu_pot_fun,'Value') == 2
    ad_options.pf_kernel = 'tria';
end
if get(handles.pop_menu_smoothing,'Value') == 1
    ad_options.pf_smoot = [0.2 1.4];
else
    p = get(handles.pop_menu_smoothing,'Value');
    p = p - 1;
    s = [0.2:0.1:1.4];
    ad_options.pf_smoot = s(p);
end

% whattodo
whattodo.bounding_box = get(handles.chk_bounding,'Value');
whattodo.bounding_box_pca = get(handles.chk_bounding_pca,'Value');
whattodo.convex_hull = get(handles.chk_convex,'Value');
whattodo.leverage = get(handles.chk_leverage,'Value');
whattodo.dist_centroid = get(handles.chk_distance_pct,'Value');
whattodo.dist_knn_fix = get(handles.chk_distance_knn,'Value');
whattodo.dist_knn_var = get(handles.chk_knn_var,'Value');
whattodo.pot_fun = get(handles.chk_pot_fun,'Value');

% -------------------------------------------------------------------------
function handles = init_handles_ad_loaded(handles)
ad_options = handles.ad_loaded.options;
whattodo = handles.ad_loaded.whattodo;
if strcmp(ad_options.pret_type,'none')
    set(handles.pop_menu_scaling,'Value',1);
elseif strcmp(ad_options.pret_type,'cent')
    set(handles.pop_menu_scaling,'Value',2);
elseif strcmp(ad_options.pret_type,'auto')
    set(handles.pop_menu_scaling,'Value',3);
elseif strcmp(ad_options.pret_type,'rang')
    set(handles.pop_menu_scaling,'Value',4); 
end
if strcmp(ad_options.distance,'euclidean')
    set(handles.pop_menu_distance,'Value',1);
elseif strcmp(ad_options.distance,'manhattan')
    set(handles.pop_menu_distance,'Value',2);
elseif strcmp(ad_options.distance,'mahalanobis')
    set(handles.pop_menu_distance,'Value',3);
end
set(handles.chk_bounding,'Value',whattodo.bounding_box);
set(handles.chk_bounding_pca,'Value',whattodo.bounding_box_pca);
set(handles.chk_convex,'Value',whattodo.convex_hull);
set(handles.chk_leverage,'Value',whattodo.leverage);
set(handles.edit_txt_leverage,'String',num2str(ad_options.lev_thr));
set(handles.chk_distance_pct,'Value',whattodo.dist_centroid);
if ad_options.dist_pct == 99;
    set(handles.pop_menu_pct,'Value',1);
elseif ad_options.dist_pct == 95;
    set(handles.pop_menu_pct,'Value',2);
elseif ad_options.dist_pct == 90;
    set(handles.pop_menu_pct,'Value',3);
elseif ad_options.dist_pct == 85;
    set(handles.pop_menu_pct,'Value',4);
else
    set(handles.pop_menu_pct,'Value',5);
end
set(handles.chk_distance_knn,'Value',whattodo.dist_knn_fix);
set(handles.pop_menu_k,'Value',ad_options.knnfix_k);
set(handles.chk_knn_var,'Value',whattodo.dist_knn_var);
p = (ad_options.knnvar_perc_test - 5)/5;
set(handles.pop_menu_validation,'Value',p);    
if ad_options.knnvar_iter == 10
    set(handles.pop_menu_iter,'Value',1);
elseif ad_options.knnvar_iter == 50
    set(handles.pop_menu_iter,'Value',2);
elseif ad_options.knnvar_iter == 100
    set(handles.pop_menu_iter,'Value',3);
elseif ad_options.knnvar_iter == 500
    set(handles.pop_menu_iter,'Value',4);
else
    set(handles.pop_menu_iter,'Value',5);
end
set(handles.pop_menu_maxk,'Value',ad_options.knnvar_k_max);
if isstr(ad_options.knnvar_k_opt)
    if strcmp(ad_options.knnvar_k_opt,'auto')
        set(handles.pop_menu_kopt,'Value',1);
    else
        set(handles.pop_menu_kopt,'Value',2);
    end
else
    set(handles.pop_menu_kopt,'Value',ad_options.knnvar_k_opt + 2);
end
set(handles.chk_pot_fun,'Value',whattodo.pot_fun);
if length(ad_options.pf_smoot) == 2
    set(handles.pop_menu_smoothing,'Value',1);
else
    s = [0.2:0.1:1.4];
    p = find(s == ad_options.pf_smoot) + 1;
    set(handles.pop_menu_smoothing,'Value',p);
end
if strcmp(ad_options.pf_kernel,'gaus')
    set(handles.pop_menu_pot_fun,'Value',1);
elseif strcmp(ad_options.pf_kernel,'tria')
    set(handles.pop_menu_pot_fun,'Value',2);
end