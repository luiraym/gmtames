function varargout = visualize_export(varargin)

% visualize_export opens a graphical interface for saving variables to the
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
                   'gui_OpeningFcn', @visualize_export_OpeningFcn, ...
                   'gui_OutputFcn',  @visualize_export_OutputFcn, ...
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


% --- Executes just before visualize_export is made visible.
function visualize_export_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;
movegui(handles.visualize_export,'center');
guidata(hObject, handles);

% set data to be saved
data_temp = varargin{1};
handles.type    = varargin{2};
handles.output = data_temp;

if strcmp (handles.type,'ad')
    set(handles.variable_name_text,'String','ad_name')
    set(handles.text_title,'String','save in matlab workspace')
elseif strcmp (handles.type,'test')
    set(handles.variable_name_text,'String','test_name')
    set(handles.text_title,'String','save in matlab workspace')
elseif strcmp (handles.type,'export')
    set(handles.variable_name_text,'String','file_name')
    set(handles.text_title,'String','export to excel')
end
guidata(hObject, handles);
uiwait(handles.visualize_export);


% --- Outputs from this function are returned to the command line.
function varargout = visualize_export_OutputFcn(hObject, eventdata, handles)
len = length(handles);
if len > 0
    varargout{1} = handles.output;
    delete(handles.visualize_export)
else
    handles.output = NaN;
    varargout{1} = handles.output;
end

% --- Executes during object creation, after setting all properties.
function variable_name_text_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

function variable_name_text_Callback(hObject, eventdata, handles)

% --- Executes on button press in save_button.
function save_button_Callback(hObject, eventdata, handles)
variable_name = get(handles.variable_name_text,'String');
if length(variable_name) > 0
    if strcmp (handles.type,'export')
        xlswrite(variable_name,handles.output);
    else
        assignin('base',variable_name,handles.output)
    end
end
guidata(hObject, handles);
uiresume(handles.visualize_export)

% --- Executes on button press in cancel_button.
function cancel_button_Callback(hObject, eventdata, handles)
guidata(hObject, handles);
uiresume(handles.visualize_export)
