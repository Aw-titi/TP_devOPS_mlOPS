const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');

const app = express();
const port = 3000;

app.use(bodyParser.json());

// Journalisation de toutes les requêtes entrantes
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  next();
});

// Route de test
app.get('/', (req, res) => {
  res.send('Bienvenue dans notre API de prédiction');
});

// Endpoint de prédiction
app.post('/predict', async (req, res) => {
  const input = req.body;
  console.log('Données reçues :', JSON.stringify(input));

  try {
    const response = await axios.post('http://localhost:8000/predict', input, {
      headers: { 'Content-Type': 'application/json' }
    });

    console.log('Réponse du modèle :', response.data);
    res.json({ prediction: response.data });

  } catch (error) {
    console.error('Erreur lors de la prédiction :', error.message);
    res.status(500).json({
      error: 'Erreur lors de la prédiction',
      details: error.response?.data || error.message
    });
  }
});

app.listen(port, '0.0.0.0', () => {
  console.log(`API en écoute sur http://localhost:${port}`);
});
