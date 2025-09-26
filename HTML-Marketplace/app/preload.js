const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  openHtmlFiles: () => ipcRenderer.invoke('open-html-files'),
  addTab: (name, content) => ipcRenderer.invoke('add-Tab', name, content),
  llmGenerate: (model, prompt) => ipcRenderer.invoke('llm-generate', { model, prompt }),
  onAddTab: (callback) => ipcRenderer.on('add-tab', (event, data) => callback(data)),
  onDefaultPages: (callback) => ipcRenderer.on('default-pages', (event, pages) => callback(pages)),
  savePersistent: (key, value) => ipcRenderer.invoke('save-persistent', key, value),
  loadPersistent: (key) => ipcRenderer.invoke('load-persistent', key),
  openHtmlFilesByPath: (paths) => ipcRenderer.invoke('open-html-files-by-path', paths),
  getAppPath: () => ipcRenderer.invoke('get-app-path'),
});