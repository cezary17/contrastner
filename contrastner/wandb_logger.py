import logging
from typing import Any, Dict

import pandas as pd
import wandb
from flair.trainers.plugins.base import TrainerPlugin

from contrastner.analysis.classification_plot import generate_classification_heatmap

log = logging.getLogger("flair")


class WandbLoggingHandler(logging.Handler):
    def __init__(self, wandb, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.wandb = wandb

    def emit(self, record):
        try:
            # adjust alert level
            if record.levelno >= logging.ERROR:
                level = self.wandb.AlertLevel.ERROR
            elif record.levelno >= logging.WARNING:
                level = self.wandb.AlertLevel.WARN
            else:
                level = self.wandb.AlertLevel.INFO

            self.wandb.alert(
                title=f"Alert from {record.module}:{record.lineno}",
                text=self.format(record),
                level=level,
            )

        except Exception:
            self.handleError(record)


class WandbLogger(TrainerPlugin):
    def __init__(self, wandb, emit_alerts=True, alert_level=logging.WARNING) -> None:
        super().__init__()

        self.wandb = wandb
        self.emit_alerts = emit_alerts
        self.alert_level = alert_level
        self._emitted_record_type_warning = False

    @TrainerPlugin.hook
    def after_setup(self, **kw):
        # a = 1
        if self.emit_alerts:
            self.log_handler = WandbLoggingHandler(self.wandb)
            self.log_handler.setLevel(self.alert_level)

            formatter = logging.Formatter("%(asctime)-15s %(message)s")
            self.log_handler.setFormatter(formatter)
            log.addHandler(self.log_handler)
        else:
            self.log_handler = None
        self.wandb.watch(self.trainer.model, log="all")

        metrics = ['micro_avg_f1-score', 'micro_avg_precision', 'micro_avg_recall', 'macro_avg_f1-score',
                   'macro_avg_precision', 'macro_avg_recall', 'accuracy', 'loss']
        for metric in metrics:
            self.wandb.define_metric(metric, summary="last")

    @TrainerPlugin.hook("_training_exception", "after_teardown")
    def close_file_handler(self, **kw):
        if self.emit_alerts:
            self.log_handler.close()
            log.removeHandler(self.log_handler)

    @TrainerPlugin.hook
    def metric_recorded(self, record):
        if record.is_scalar_list:
            self.wandb.log({record.joined_name: str(record.value[0])})
            if len(record.value) > 1:
                log.warning(
                    f"Only the first element of the scalar list {record.joined_name} is logged to wandb."
                )
        elif record.is_scalar:
            self.wandb.log({record.joined_name: str(record.value)})
        elif record.is_string:
            self.wandb.log({record.joined_name: record.value})

    @TrainerPlugin.hook
    def after_training(self, **kw):
        results = self.trainer.return_values

        # log simple score
        self.wandb.log({"test_score": results["test_score"]})

        # log classification report as a wandb table
        try:
            # As table
            classification_report = pd.DataFrame(results["test_results"].classification_report).reset_index().rename(
                columns={"index": "metric"})
            classification_table = wandb.Table(dataframe=classification_report)
            self.wandb.log({"classification_report": classification_table})
            # As heatmap
            heatmap_image = generate_classification_heatmap(results["test_results"].classification_report)
            self.wandb.log({"classification_heatmap": wandb.Image(heatmap_image)})

            metric_logs = {self.format_output_key(key): val for key, val in results["test_results"].scores.items()}
            self.wandb.log(metric_logs)
        except KeyError:
            log.info("No test_results available, skipping classification report")

        try:
            # log contrast label statistics
            contrast_label_statistics = self.model.label_statistics
            anchor_label_statistics = contrast_label_statistics["anchors"]
            negative_label_statistics = contrast_label_statistics["negatives"]

            labels = list(negative_label_statistics.keys())  # use this to include O label

            # add combined statistics
            labels.append("combined_no_o")
            anchor_label_statistics["combined_no_o"] = sum([v for k, v in anchor_label_statistics.items() if k != "O"])
            negative_label_statistics["combined_no_o"] = sum(
                [v for k, v in negative_label_statistics.items() if k != "O"])

            df = pd.DataFrame({
                "label": labels,
                "anchor_count": [anchor_label_statistics[label] for label in labels],
                "negative_count": [negative_label_statistics[label] for label in labels]
            })

            label_statistics_table = wandb.Table(dataframe=df)
            self.wandb.log({"contrast_label_statistics": label_statistics_table})

        except AttributeError:
            log.info("No contrast_label_statistics available, skipping contrast label statistics")

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "emit_alerts": self.emit_alerts,
            "alert_level": self.alert_level,
        }

    @staticmethod
    def format_output_key(key):
        if isinstance(key, tuple):
            # Join tuple elements with underscore and replace spaces
            return "_".join(str(item).replace(" ", "_") for item in key)
        else:
            # For non-tuple keys, just replace spaces with underscores
            return str(key).replace(" ", "_")
