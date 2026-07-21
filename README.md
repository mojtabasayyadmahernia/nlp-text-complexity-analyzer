## Live Demo

The API is deployed on **Render.com** (Frankfurt, EU  ).

**Base URL:** `https://nlp-text-complexity-analyzer.onrender.com`

**Interactive Documentation (Swagger UI  ):** [https://nlp-text-complexity-analyzer.onrender.com/docs](https://nlp-text-complexity-analyzer.onrender.com/docs  )

> Note: The service uses Render's free tier and may take 30–60 seconds to wake up after a period of inactivity.

### Quick Test

```bash
curl -X POST "https://nlp-text-complexity-analyzer.onrender.com/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Paste your text here (minimum 50 characters)."}'
