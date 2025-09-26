const fs = require('fs');
const path = require('path');

const DATA_PATH = path.join(__dirname, 'user-data.json');

function loadData() {
  if (fs.existsSync(DATA_PATH)) {

    try {
      return JSON.parse(fs.readFileSync(DATA_PATH, 'utf-8'));

    } catch {
      return {};
      
    }
  }
  return {};
}

function saveData(data) {
  fs.writeFileSync(DATA_PATH, JSON.stringify(data, null, 2), 'utf-8');
}

module.exports = { loadData, saveData, DATA_PATH };