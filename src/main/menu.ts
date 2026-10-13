import { app, shell, BrowserWindow, Menu, MenuItem } from 'electron';

const REPO_URL = 'https://github.com/gerp93/KVGenius';
const ISSUES_URL = 'https://github.com/gerp93/KVGenius/issues';

/**
 * Per gerp93/KVG_Standards' Electron application menu standard: no relying on
 * Electron's default File/Edit/View/Window/Help menu bar. Keep only View
 * (dev tools gated behind !app.isPackaged, zoom, fullscreen) and Help
 * (repo link, issues link, version), plus the macOS-only app-name menu.
 * Dropping Edit means Cut/Copy/Paste/Select All need attachContextMenu()
 * below to stay reachable by right-click.
 */
export function setupApplicationMenu(): void {
  const isMac = process.platform === 'darwin';

  const viewMenu: Electron.MenuItemConstructorOptions = {
    label: 'View',
    submenu: [
      ...(!app.isPackaged
        ? [
            { role: 'reload' as const },
            { role: 'forceReload' as const },
            { role: 'toggleDevTools' as const },
            { type: 'separator' as const },
          ]
        : []),
      { role: 'zoomIn' as const },
      { role: 'zoomOut' as const },
      { role: 'resetZoom' as const },
      { type: 'separator' as const },
      { role: 'togglefullscreen' as const },
    ],
  };

  const helpMenu: Electron.MenuItemConstructorOptions = {
    label: 'Help',
    role: 'help',
    submenu: [
      { label: 'GitHub Repository', click: () => void shell.openExternal(REPO_URL) },
      { label: 'Report an Issue', click: () => void shell.openExternal(ISSUES_URL) },
      { type: 'separator' as const },
      { label: `Version ${app.getVersion()}`, enabled: false },
    ],
  };

  const template: Electron.MenuItemConstructorOptions[] = [
    ...(isMac
      ? [
          {
            label: app.name,
            submenu: [
              { role: 'about' as const },
              { type: 'separator' as const },
              { role: 'services' as const },
              { type: 'separator' as const },
              { role: 'hide' as const },
              { role: 'hideOthers' as const },
              { role: 'unhide' as const },
              { type: 'separator' as const },
              { role: 'quit' as const },
            ],
          },
        ]
      : []),
    viewMenu,
    helpMenu,
  ];

  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

/** Right-click in a text box: spelling suggestions and "Add to dictionary" for a misspelled word (underlined by the
 * built-in spell checker), then Cut/Copy/Paste/Select All, since the Edit menu was dropped above. */
export function attachContextMenu(win: BrowserWindow): void {
  win.webContents.on('context-menu', (_event, params) => {
    if (!params.isEditable) return;

    const menu = new Menu();
    for (const suggestion of params.dictionarySuggestions) {
      menu.append(new MenuItem({ label: suggestion, click: () => win.webContents.replaceMisspelling(suggestion) }));
    }
    if (params.misspelledWord) {
      const word = params.misspelledWord;
      menu.append(new MenuItem({ label: 'Add to dictionary', click: () => win.webContents.session.addWordToSpellCheckerDictionary(word) }));
    }
    if (menu.items.length > 0) menu.append(new MenuItem({ type: 'separator' }));
    if (params.editFlags.canCut) menu.append(new MenuItem({ role: 'cut' }));
    if (params.editFlags.canCopy) menu.append(new MenuItem({ role: 'copy' }));
    if (params.editFlags.canPaste) menu.append(new MenuItem({ role: 'paste' }));
    if (params.editFlags.canSelectAll) menu.append(new MenuItem({ role: 'selectAll' }));

    if (menu.items.length > 0) menu.popup();
  });
}
