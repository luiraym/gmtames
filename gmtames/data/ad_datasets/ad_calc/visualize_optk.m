function varargout = visualize_optk(varargin)

% visualize_optk opens a graphical interface for setting the optimal k
% value in the kNN with variable k approach
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
                   'gui_OpeningFcn', @visualize_optk_OpeningFcn, ...
                   'gui_OutputFcn',  @visualize_optk_OutputFcn, ...
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


% --- Executes just before visualize_optk is made visible.
function visualize_optk_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
movegui('center');
handles.k_optimisation = varargin{1};
% init combo
str_disp={};
for j=1:size(handles.k_optimisation,2)
    str_disp{j} = num2str(j);
end
set(handles.pop_optk,'String',str_disp);
set(handles.pop_optk,'Value',1);
guidata(hObject, handles);
uiwait(handles.visualize_optk);

% --- Outputs from this function are returned to the command line.
function varargout = visualize_optk_OutputFcn(hObject, eventdata, handles)
len = length(handles);
if len > 0
    varargout{1} = handles.k_opt;
    delete(handles.visualize_optk)
else
    varargout{1} = 1;
end

% --- Executes on button press in button_ok.
function button_ok_Callback(hObject, eventdata, handles)
handles.k_opt = get(handles.pop_optk,'Value');
guidata(hObject,handles)
uiresume(handles.visualize_optk)
    
% --- Executes during object creation, after setting all properties.
function pop_optk_CreateFcn(hObject, eventdata, handles)
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% --- Executes on selection change in pop_optk.
function pop_optk_Callback(hObject, eventdata, handles)
