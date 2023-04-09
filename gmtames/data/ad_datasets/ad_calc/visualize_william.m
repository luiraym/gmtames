function varargout = visualize_william(varargin)

% visualize_william opens a graphical interface for loading variables from the
% workspace for the william plot
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
                   'gui_OpeningFcn', @visualize_william_OpeningFcn, ...
                   'gui_OutputFcn',  @visualize_william_OutputFcn, ...
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

% --- Executes just before visualize_william is made visible.
function visualize_william_OpeningFcn(hObject, eventdata, handles, varargin)

movegui(handles.visualize_william,'center');
handles.data = varargin{1};

% enable/disable buttons and menu
handles = enable_disable(handles);

% updtae list box
update_listbox(handles)

guidata(hObject, handles);
uiwait(handles.visualize_william);

% --- Outputs from this function are returned to the command line.
function varargout = visualize_william_OutputFcn(hObject, eventdata, handles)
len = length(handles);
if len > 0
    varargout{1} = handles.data;
    varargout{2} = handles.dowilliam;
    varargout{3} = get(handles.chk_labels,'Value');
    delete(handles.visualize_william)
else
    handles.data = NaN;
    handles.dowilliam = 0;
    varargout{1} = handles.data;
    varargout{2} = handles.dowilliam;
    varargout{3} = get(handles.chk_labels,'Value');
end

% --- Executes on button press in button_train_exp.
function button_train_exp_Callback(hObject, eventdata, handles)
res = visualize_load(3,size(handles.data.Xtrain));
if isnan(res.loaded_file)
    % do nothing
elseif res.from_file == 1
    tmp_data = load(res.path);
    handles.data.y_train_exp = getfield(tmp_data,res.name);
    if size(handles.data.y_train_exp,2) > size(handles.data.y_train_exp,1)
        handles.data.y_train_exp = handles.data.y_train_exp';
    end
    handles.data.y_train_exp_name = res.name;
else
    handles.data.y_train_exp = evalin('base',res.name);
    if size(handles.data.y_train_exp,2) > size(handles.data.y_train_exp,1)
        handles.data.y_train_exp = handles.data.y_train_exp';
    end
    handles.data.y_train_exp_name = res.name;
end
update_listbox(handles)
handles = enable_disable(handles);
guidata(hObject, handles);

% --- Executes on button press in button_train_calc.
function button_train_calc_Callback(hObject, eventdata, handles)
res = visualize_load(3,size(handles.data.Xtrain));
if isnan(res.loaded_file)
    % do nothing
elseif res.from_file == 1
    tmp_data = load(res.path);
    handles.data.y_train_calc = getfield(tmp_data,res.name);
    if size(handles.data.y_train_calc,2) > size(handles.data.y_train_calc,1)
        handles.data.y_train_calc = handles.data.y_train_calc';
    end
    handles.data.y_train_calc_name = res.name;
else
    handles.data.y_train_calc = evalin('base',res.name);
    if size(handles.data.y_train_calc,2) > size(handles.data.y_train_calc,1)
        handles.data.y_train_calc = handles.data.y_train_calc';
    end
    handles.data.y_train_calc_name = res.name;
end
update_listbox(handles)
handles = enable_disable(handles);
guidata(hObject, handles);

% --- Executes on button press in button_test_exp.
function button_test_exp_Callback(hObject, eventdata, handles)
res = visualize_load(3,size(handles.data.Xtest));
if isnan(res.loaded_file)
    % do nothing
elseif res.from_file == 1
    tmp_data = load(res.path);
    handles.data.y_test_exp = getfield(tmp_data,res.name);
    if size(handles.data.y_test_exp,2) > size(handles.data.y_test_exp,1)
        handles.data.y_test_exp = handles.data.y_test_exp';
    end
    handles.data.y_test_exp_name = res.name;
else
    handles.data.y_test_exp = evalin('base',res.name);
    if size(handles.data.y_test_exp,2) > size(handles.data.y_test_exp,1)
        handles.data.y_test_exp = handles.data.y_test_exp';
    end
    handles.data.y_test_exp_name = res.name;
end
update_listbox(handles)
handles = enable_disable(handles);
guidata(hObject, handles);

% --- Executes on button press in button_test_calc.
function button_test_calc_Callback(hObject, eventdata, handles)
res = visualize_load(3,size(handles.data.Xtest));
if isnan(res.loaded_file)
    % do nothing
elseif res.from_file == 1
    tmp_data = load(res.path);
    handles.data.y_test_calc = getfield(tmp_data,res.name);
    if size(handles.data.y_test_calc,2) > size(handles.data.y_test_calc,1)
        handles.data.y_test_calc = handles.data.y_test_calc';
    end
    handles.data.y_test_calc_name = res.name;
else
    handles.data.y_test_calc = evalin('base',res.name);
    if size(handles.data.y_test_calc,2) > size(handles.data.y_test_calc,1)
        handles.data.y_test_calc = handles.data.y_test_calc';
    end
    handles.data.y_test_calc_name = res.name;
end
update_listbox(handles)
handles = enable_disable(handles);
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function list_box_response_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% --- Executes on selection change in list_box_response.
function list_box_response_Callback(hObject, eventdata, handles)

% --- Executes on button press in button_calculate.
function button_calculate_Callback(hObject, eventdata, handles)
handles.dowilliam = 1;
guidata(hObject,handles)
uiresume(handles.visualize_william)
    
% --- Executes on button press in button_cancel.
function button_cancel_Callback(hObject, eventdata, handles)
handles.dowilliam = 0;
guidata(hObject,handles)
uiresume(handles.visualize_william)

% ------------------------------------------------------------------------
function handles = enable_disable(handles)
if length(handles.data.y_train_exp) == 0 | length(handles.data.y_train_calc) == 0 | length(handles.data.y_test_exp) == 0 | length(handles.data.y_test_calc) == 0
    set(handles.button_calculate,'Enable','off');
else
    set(handles.button_calculate,'Enable','on');
end

% ------------------------------------------------------------------------
function update_listbox(handles)
str{1} = ['Training set: ' num2str(size(handles.data.Xtrain,1)) ' samples'];
str{2} = ['Test set: ' num2str(size(handles.data.Xtest,1)) ' samples'];
if length(handles.data.y_train_exp) == 0
    str{3} = ['training experimental response: not loaded'];
else
    str{3} = ['training experimental response: ' handles.data.y_train_exp_name ' (' num2str(length(handles.data.y_train_exp)) ' x 1)'];
end
if length(handles.data.y_train_calc) == 0
    str{4} = ['training calculated response: not loaded'];
else
    str{4}= ['training calculated response: ' handles.data.y_train_calc_name ' (' num2str(length(handles.data.y_train_calc)) ' x 1)'];
end
if length(handles.data.y_test_exp) == 0
    str{5} = ['test experimental response: not loaded'];
else
    str{5} = ['test experimental response: ' handles.data.y_test_exp_name ' (' num2str(length(handles.data.y_test_exp)) ' x 1)'];
end
if length(handles.data.y_test_calc) == 0
    str{6} = ['test calculated response: not loaded'];
else
    str{6} = ['test calculated response: ' handles.data.y_test_calc_name ' (' num2str(length(handles.data.y_test_calc)) ' x 1)'];
end
set(handles.list_box_response,'String',str);


% --- Executes on button press in chk_labels.
function chk_labels_Callback(hObject, eventdata, handles)
% hObject    handle to chk_labels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of chk_labels


