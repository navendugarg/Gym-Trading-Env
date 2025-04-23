from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import pandas as pd
import json
from pathlib import Path
import glob
from .utils.charts import charts  # Assuming charts returns a Plotly figure

class Renderer:
    """Handles rendering of trading data visualizations using FastAPI and Plotly."""

    def __init__(self, render_logs_dir: str):
        """Initialize the Renderer with the directory containing render logs."""
        self.app = FastAPI()
        self.templates = Jinja2Templates(directory="templates")
        self.render_logs_dir = render_logs_dir
        self.metrics = []
        self.lines = []

    def add_metric(self, name: str, function: callable) -> None:
        """Add a custom metric to be computed and displayed."""
        self.metrics.append({"name": name, "function": function})

    def add_line(self, name: str, function: callable, line_options: dict = None) -> None:
        """Add a custom line to the chart."""
        line = {"name": name, "function": function}
        if line_options is not None:
            line["line_options"] = line_options
        self.lines.append(line)

    def load_data(self, name: str = None) -> pd.DataFrame:
        """Load data from the specified render log or the latest one if not specified."""
        if not name:
            render_pathes = glob.glob(f"{self.render_logs_dir}/*.pkl")
            if not render_pathes:
                raise HTTPException(status_code=404, detail="No data available")
            name = Path(render_pathes[-1]).name
        try:
            return pd.read_pickle(f"{self.render_logs_dir}/{name}")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="File not found")

    def compute_metrics(self, df: pd.DataFrame) -> list:
        """Compute all metrics for the given DataFrame."""
        return [{"name": m["name"], "value": m["function"](df)} for m in self.metrics]

    def run(self) -> None:
        """Run the FastAPI application."""
        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            render_pathes = glob.glob(f"{self.render_logs_dir}/*.pkl")
            render_names = [Path(path).name for path in render_pathes]
            return self.templates.TemplateResponse("index.html", {"request": request, "render_names": render_names})

        @self.app.get("/update_data")
        async def update(name: str = None):
            df = self.load_data(name)
            chart = charts(df, self.lines)
            return JSONResponse(content=json.loads(chart.to_json()))

        @self.app.get("/metrics")
        async def get_metrics(name: str = None):
            df = self.load_data(name)
            metrics = self.compute_metrics(df)
            return JSONResponse(content=metrics)

        uvicorn.run(self.app, host="0.0.0.0", port=8000)