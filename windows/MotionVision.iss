#ifndef AppVersion
#define AppVersion "1.0.0"
#endif

[Setup]
AppId={{A8DF4969-3EC3-4D31-96FC-85CE4DBAA2B7}
AppName=Motion Vision
AppVersion={#AppVersion}
AppPublisher=Motion Vision
DefaultDirName={autopf}\Motion Vision
DefaultGroupName=Motion Vision
OutputDir=..\dist\windows
OutputBaseFilename=MotionVision-Setup
SetupIconFile=..\assets\icon\MotionVision.ico
UninstallDisplayIcon={app}\MotionVision.exe
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "..\dist\windows\MotionVision.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\config\.env.example"; DestDir: "{app}\config"; Flags: ignoreversion

[Icons]
Name: "{group}\Motion Vision"; Filename: "{app}\MotionVision.exe"; IconFilename: "{app}\MotionVision.exe"
Name: "{autodesktop}\Motion Vision"; Filename: "{app}\MotionVision.exe"; Tasks: desktopicon; IconFilename: "{app}\MotionVision.exe"

[Run]
Filename: "{app}\MotionVision.exe"; Description: "Launch Motion Vision"; Flags: postinstall nowait skipifsilent
