
# Smart Sort

Smart Sort records waste disposal recommendations and provides an operations dashboard for reviewing transaction volume, response times, and input methods.

## Project layout

```text
app/                    Streamlit dashboard implementation
barcode/                Barcode scanner prototype
ml/                     Image classification and training files
tests/                  Server tests
server.py               Local scan receiver and SQLite writer ( using 
```

The dashboard lets you choose between generated sample data and the live SQLite database. Sample data is the default preview and is not written to the database. Live database mode shows its connection state and an empty state if no transactions are available.

## Run the dashboard

From the project root:

```bash
streamlit run app/dashboard.py
```

`app/dashboard.py` is the only dashboard. Set `SMARTSORT_DB` to use a different SQLite database path.

## Run the scan receiver

In a separate terminal, from the project root:

```bash
python server.py
```

The receiver listens on port `8000` by default. It accepts completed barcode lookups at `/api/scan` and stores classified transactions in the same database used by the dashboard. `SMARTSORT_HOST`, `SMARTSORT_PORT`, and `SMARTSORT_DB` can override its defaults.

The image classification, training scripts, datasets, and model weights remain in `ml/`.

### Download Requirements

```bash
python -m pip install -r server/requirements.txt
```

#### Including

- **fastapi** for the API
- **pydantic** for request validation
- **uvicorn**  to run the server API