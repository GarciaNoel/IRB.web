const { app, BrowserWindow, dialog, ipcMain } = require('electron')
const fs = require('fs')
const path = require('path')
const { loadData, saveData } = require('./storage.js');

const DEFAULT_PAGES_PATH = path.join(__dirname, 'defaultPages.json');

function loadDefaultPages() {
  if (fs.existsSync(DEFAULT_PAGES_PATH)) {
    return JSON.parse(fs.readFileSync(DEFAULT_PAGES_PATH, 'utf-8'));

  }
  return [
    path.join(__dirname, 'page1.html'),
    path.join(__dirname, 'page2.html'),
    path.join(__dirname, 'page3.html')
  ];
}

function saveDefaultPages(pages) {
  fs.writeFileSync(DEFAULT_PAGES_PATH, JSON.stringify(pages, null, 2), 'utf-8');
}

let defaultPages = loadDefaultPages();
let persistentData = loadData();

function createWindow() {
  const win = new BrowserWindow({
    width: 1000,
    height: 700,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  })

  win.setMenu(null);
  win.loadFile('app/index.html')

  //win.webContents.openDevTools();

  ipcMain.handle('save-persistent', (event, key, value) => {
    persistentData[key] = value;
    saveData(persistentData);
  });

  ipcMain.handle('load-persistent', (event, key) => {
    return persistentData[key];
  });

  ipcMain.handle('get-app-path', () => {
    return __dirname;
  });

  win.webContents.on('did-finish-load', () => {
    const pages = defaultPages
      .filter(fp => fs.existsSync(fp))
      .map(fp => ({
        name: path.basename(fp),
        content: fs.readFileSync(fp, 'utf-8')
      }));
    win.webContents.send('default-pages', pages);
  });

  ipcMain.handle('open-html-files', async () => {
    const { canceled, filePaths } = await dialog.showOpenDialog(win, {
      filters: [{ name: 'HTML Files', extensions: ['html', 'htm'] }],
      properties: ['openFile', 'multiSelections']
    });

    if (canceled || filePaths.length === 0) return [];
    let updated = false;

    for (const filePath of filePaths) {

      if (!defaultPages.includes(filePath)) {
        defaultPages.push(filePath);
        updated = true;

      }
    }

    if (updated) saveDefaultPages(defaultPages);

    return filePaths.map(filePath => ({
      name: path.basename(filePath),
      content: fs.readFileSync(filePath, 'utf-8')
    }));
  });

  ipcMain.handle('add-Tab', (event, name, content) => {
    const win = BrowserWindow.getFocusedWindow();

    if (win) {
      win.webContents.send('add-tab', { name, content });
      win.loadFile('app/index.html')

    }
  });

  ipcMain.handle('open-html-files-by-path', async (event, filePaths) => {
    if (!filePaths || filePaths.length === 0) return [];
    const paths = Array.isArray(filePaths) ? filePaths : [filePaths];

    let updated = false;
    for (const filePath of paths) {
      const absPath = path.isAbsolute(filePath) ? filePath : path.join(__dirname, filePath);
      if (!defaultPages.includes(absPath)) {
        defaultPages.push(absPath);
        updated = true;
      }
    }
    if (updated) saveDefaultPages(defaultPages);

    return paths
      .map(filePath => path.isAbsolute(filePath) ? filePath : path.join(__dirname, filePath))
      .filter(filePath => fs.existsSync(filePath))
      .map(filePath => ({
        name: path.basename(filePath),
        content: fs.readFileSync(filePath, 'utf-8')
      }));
  });

  const http = require('http'); 

  ipcMain.handle('llm-generate', async (event, { model, prompt }) => {
    return new Promise((resolve, reject) => {
      const data = JSON.stringify({
        model,
        prompt,
        stream: false
      });

      const req = http.request({
        hostname: 'localhost',
        port: 11434,
        path: '/api/generate',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(data)
        }
      }, res => {
        let body = '';
        res.on('data', chunk => body += chunk);
        res.on('end', () => {
          try {
            const json = JSON.parse(body);
            resolve(json.response || body);

          } catch (e) {
            resolve(body);
            
          }
        });
      });

      req.on('error', reject);
      req.write(data);
      req.end();
    });
  });
}

app.whenReady().then(createWindow)

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow()
})