function varargout = visualize_load(varargin)

% visualize_load opens a graphical interface for loading variables from the
% workspace
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
                   'gui_OpeningFcn', @visualize_load_OpeningFcn, ...
                   'gui_OutputFcn',  @visualize_load_OutputFcn, ...
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


% --- Executes just before visualize_load is made visible.
function visualize_load_OpeningFcn(hObject, eventdata, handles, varargin)

%handles.output = hObject;
movegui(handles.visualize_load_form,'center');
if varargin{1} == 1
    handles.load_type = 'train';
elseif varargin{1} == 2
    handles.load_type = 'test';    
elseif varargin{1} == 3
    handles.load_type = 'response';    
elseif varargin{1} == 4
    handles.load_type = 'model'; 
elseif varargin{1} == 5
    handles.load_type = 'sample_labels';     
elseif varargin{1} == 6
    handles.load_type = 'variable_labels';     
end

handles.num_samples = varargin{2}(1);
handles.num_variables = varargin{2}(2);
handles.output.loaded_file = NaN;
handles.output.from_file = 0;

% update listbox
vars = evalin('base','whos');
handles = update_listbox(handles,vars);

guidata(hObject, handles);
uiwait(handles.visualize_load_form);

% --- Outputs from this function are returned to the command line.
function varargout = visualize_load_OutputFcn(hObject, eventdata, handles)
len = length(handles);
if len > 0
    varargout{1} = handles.output;
    delete(handles.visualize_load_form)
else
    handles.output.loaded_file = NaN;
    varargout{1} = handles.output;
end

% --- Executes during object creation, after setting all properties.
function listbox_variables_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% --- Executes on selection change in listbox_variables.
function listbox_variables_Callback(hObject, eventdata, handles)

% --- Executes on button press in button_load.
function button_load_Callback(hObject, eventdata, handles)
if length(handles.vars_listed) == 0
    handles.output.loaded_file = NaN;
    guidata(hObject,handles)
    uiresume(handles.visualize_load_form)
else
    errortype = 'none';
    if strcmp(handles.load_type,'response')
        errortype = check_response(handles);
    elseif strcmp(handles.load_type,'model')
        errortype = check_model(handles);
    elseif strcmp(handles.load_type,'sample_labels')
        errortype = check_sample_labels(handles);
    elseif strcmp(handles.load_type,'variable_labels')
        errortype = check_variable_labels(handles);
    elseif strcmp(handles.load_type,'test')
        errortype = check_test(handles);
    end
    if strcmp(errortype,'none')
        in = get(handles.listbox_variables,'Value');
        handles.output.name = handles.vars_listed(in).name;
        handles.output.loaded_file = 1;
        guidata(hObject,handles)
        uiresume(handles.visualize_load_form)
    else
        errordlg(errortype,'loading error') 
    end
end

% --- Executes on button press in button_cancel.
function button_cancel_Callback(hObject, eventdata, handles)
handles.output.loaded_file = NaN;
guidata(hObject,handles)
uiresume(handles.visualize_load_form)

% --- Executes on button press in button_load_from_file.
function button_load_from_file_Callback(hObject, eventdata, handles)
[FileName,PathName] = uigetfile('*.mat','Select mat-file');
if isstr(FileName)
    vars = whos('-file',[PathName FileName]);
    handles = update_listbox(handles,vars);
    handles.output.path = [PathName FileName];
    handles.output.from_file = 1;
    guidata(hObject,handles)
end

% -------------------------------------------------------------------------
function handles = update_listbox(handles,vars)

% filter type of data to be loaded
set(handles.listbox_variables,'Value',1);
cnt = 1;
vars_are_listed = 0;
if strcmp(handles.load_type,'train') | strcmp(handles.load_type,'test')
    for k = 1:length(vars)
        if strcmp (vars(k).class,'double')
            if (vars(k).size(2) > 1 & vars(k).size(1) > 1) & length(vars(k).size) == 2
                label_vars_in{cnt} = [vars(k).name '     [' num2str(vars(k).size(1)) 'x' num2str(vars(k).size(2)) ']     ' vars(k).class];
                vars_listed(cnt) = vars(k);
                vars_are_listed = 1;
                cnt = cnt + 1;
            end
        end
    end
elseif strcmp(handles.load_type,'response')
    for k = 1:length(vars)
        if strcmp (vars(k).class,'double')
            if (vars(k).size(2) == 1 | vars(k).size(1) == 1)
                label_vars_in{cnt} = [vars(k).name '     [' num2str(vars(k).size(1)) 'x' num2str(vars(k).size(2)) ']     ' vars(k).class];
                vars_listed(cnt) = vars(k);
                vars_are_listed = 1;
                cnt = cnt + 1;
            end
        end
    end
elseif strcmp(handles.load_type,'model')
    for k = 1:length(vars)
        if strcmp (vars(k).class,'struct')
            if (vars(k).size(2) == 1 | vars(k).size(1) == 1)
                label_vars_in{cnt} = [vars(k).name '     [' num2str(vars(k).size(1)) 'x' num2str(vars(k).size(2)) ']     ' vars(k).class];
                vars_listed(cnt) = vars(k);
                vars_are_listed = 1;
                cnt = cnt + 1;
            end
        end
    end    
elseif strcmp(handles.load_type,'sample_labels')
    for k = 1:length(vars)
        if strcmp (vars(k).class,'cell')
            if (vars(k).size(2) == 1 | vars(k).size(1) == 1)
                label_vars_in{cnt} = [vars(k).name '     [' num2str(vars(k).size(1)) 'x' num2str(vars(k).size(2)) ']     ' vars(k).class];
                vars_listed(cnt) = vars(k);
                vars_are_listed = 1;
                cnt = cnt + 1;
            end
        end
    end
elseif strcmp(handles.load_type,'variable_labels')
    for k = 1:length(vars)
        if strcmp (vars(k).class,'cell')
            if (vars(k).size(2) == 1 | vars(k).size(1) == 1)
                label_vars_in{cnt} = [vars(k).name '     [' num2str(vars(k).size(1)) 'x' num2str(vars(k).size(2)) ']     ' vars(k).class];
                vars_listed(cnt) = vars(k);
                vars_are_listed = 1;
                cnt = cnt + 1;
            end
        end
    end  
end
if vars_are_listed
    set(handles.listbox_variables,'String',label_vars_in);
    handles.vars_listed = vars_listed;
else
    set(handles.listbox_variables,'String','no allowed variables in selected workspace');
    handles.vars_listed = [];
end

% -------------------------------------------------------------------------
function errortype = check_response(handles)
errortype = 'none';
if handles.output.from_file == 1
    tmp_data = load(handles.output.path);
    in = get(handles.listbox_variables,'Value');
    response = getfield(tmp_data,handles.vars_listed(in).name);
    if size(response,2) > size(response,1)
        response = response';
    end
else
    in = get(handles.listbox_variables,'Value');
    response = evalin('base',handles.vars_listed(in).name);
    if size(response,2) > size(response,1)
        response = response';
    end
end
% no. of samples for data and response
if size(response,1) ~= handles.num_samples
    chk = 0;
    errortype = 'input error: data and response must have the same number of rows';
    return
end

% -------------------------------------------------------------------------
function errortype = check_model(handles)
errortype = 'none';
if handles.output.from_file == 1
    tmp_data = load(handles.output.path);
    in = get(handles.listbox_variables,'Value');
    model = getfield(tmp_data,handles.vars_listed(in).name);
else
    in = get(handles.listbox_variables,'Value');
    model = evalin('base',handles.vars_listed(in).name);
end
% model is a toolbox structure 
if ~isfield(model,'exp_var') | ~isfield(model,'cum_var') | ~isfield(model,'E') | ~isfield(model,'L')
    errortype = 'input error: only PCA models can be loaded. The selected structure is not recognized as a PCA model created by this toolbox';
    return
end

% -------------------------------------------------------------------------
function errortype = check_test(handles)
errortype = 'none';
if handles.output.from_file == 1
    tmp_data = load(handles.output.path);
    in = get(handles.listbox_variables,'Value');
    Xtest = getfield(tmp_data,handles.vars_listed(in).name);
else
    in = get(handles.listbox_variables,'Value');
    Xtest = evalin('base',handles.vars_listed(in).name);
end
% variables in test and trainign are different 
if handles.num_variables ~= size(Xtest,2)                                     
    errortype = 'input error: training and test sets must have the same number of variables';
    return
end

% -------------------------------------------------------------------------
function errortype = check_sample_labels(handles)
errortype = 'none';
if handles.output.from_file == 1
    tmp_data = load(handles.output.path);
    in = get(handles.listbox_variables,'Value');
    labels = getfield(tmp_data,handles.vars_listed(in).name);
else
    in = get(handles.listbox_variables,'Value');
    labels = evalin('base',handles.vars_listed(in).name);
end
% settings is a toolbox structure
if length(labels) ~= handles.num_samples
    errortype = 'input error: sample labels must be structured as cell array with a number of elements equal to the number of samples';
end

% -------------------------------------------------------------------------
function errortype = check_variable_labels(handles)
errortype = 'none';
if handles.output.from_file == 1
    tmp_data = load(handles.output.path);
    in = get(handles.listbox_variables,'Value');
    labels = getfield(tmp_data,handles.vars_listed(in).name);
else
    in = get(handles.listbox_variables,'Value');
    labels = evalin('base',handles.vars_listed(in).name);
end
% settings is a toolbox structure
if length(labels) ~= handles.num_samples
    errortype = 'input error: variable labels must be structured as cell array with a number of elements equal to the number of variables';
end