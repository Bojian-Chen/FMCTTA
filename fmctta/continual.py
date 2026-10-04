"""Timestamp-ordered fault-onset simulation and causal online reports."""
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch

from .data import FaultDomainDataset


def parse_class_onsets(value, num_classes):
    """Seconds relative to the selected domain start; null excludes a class."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    values = value.split(',') if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)) or len(values) != num_classes:
        raise ValueError('continual_class_onsets must contain one entry per class')
    result = []
    for item in values:
        if item is None or str(item).strip().lower() in {'none', 'null'}:
            result.append(None)
            continue
        onset = float(item)
        if not np.isfinite(onset) or onset < 0:
            raise ValueError('continual_class_onsets must be finite nonnegative seconds')
        result.append(onset)
    if result[0] != 0 or any(t == 0 for t in result[1:] if t is not None):
        raise ValueError('continual requires class 0 at zero and faults strictly after zero')
    if all(t is None for t in result[1:]):
        raise ValueError('continual requires at least one fault onset')
    return result


def build_continual_stream(dataset, window_start_sec, num_classes, onsets=None):
    if window_start_sec is None:
        raise ValueError('continual requires MAT window_start_sec; rebuild with ctta_data_builder')
    ids = dataset.sample_ids.numpy()
    labels = dataset.labels.numpy()
    times = np.asarray(window_start_sec, dtype=float).reshape(-1)[ids]
    if not len(times) or not np.isfinite(times).all():
        raise ValueError('continual requires nonempty, finite window timestamps')
    origin = float(times.min())
    elapsed = times - origin
    duration = float(elapsed.max())
    onsets = parse_class_onsets(onsets, num_classes)
    if onsets is None:
        onsets = [0.0] + [duration * c / num_classes for c in range(1, num_classes)]
        onsets = parse_class_onsets(onsets, num_classes)
    keep = np.zeros(len(ids), dtype=bool)
    for class_id, onset in enumerate(onsets):
        if onset is None:
            continue
        selected = (labels == class_id) & (elapsed >= onset)
        if not selected.any():
            raise ValueError(f'continual class {class_id} has no windows at/after {onset:g}s')
        keep |= selected
    indices = np.flatnonzero(keep)
    indices = indices[np.lexsort((ids[indices], labels[indices], times[indices]))]
    if labels[indices[0]] != 0:
        raise ValueError('continual stream must start with healthy class 0')
    elapsed = elapsed[indices]
    labels = labels[indices]
    events = []
    for class_id, onset in enumerate(onsets):
        if onset is not None:
            first = int(np.flatnonzero(labels == class_id)[0])
            events.append(dict(class_id=class_id, onset_sec=onset,
                               first_stream_index=first + 1,
                               first_sample_time_sec=float(elapsed[first])))
    index = torch.as_tensor(indices, dtype=torch.long)
    stream = FaultDomainDataset(dataset.signals[index], dataset.labels[index],
                                dataset.sample_ids[index], dataset.domain)
    metadata = dict(time_origin_sec=origin, duration_sec=duration,
                    class_onsets_sec=onsets, events=events,
                    candidate_samples=len(ids), emitted_samples=len(indices),
                    discarded_before_onset_or_excluded=int((~keep).sum()),
                    emitted_samples_per_class=np.bincount(labels, minlength=num_classes).tolist(),
                    construction='independent recordings; active classes interleaved by original timestamp',
                    time_unit='seconds relative to domain start')
    return stream, elapsed, metadata


def load_operating_condition(manifest_path, domain, domain_name):
    """Read the builder's nominal schedule, without claiming measured telemetry."""
    result = dict(domain=int(domain), domain_name=domain_name,
                  definition='Nominal profile at window start; not measured telemetry')
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        result['phases'] = manifest.get('condition_profile', [])
    return result


def operating_condition_values(times, condition):
    name = condition.get('domain_name', '')
    torque = re.fullmatch(r'torque_(\d+)rpm_(\d+)to(\d+)Nm__.*', name)
    speed = re.fullmatch(r'speed_(\d+)Nm_(\d+)to(\d+)rpm__.*', name)
    match = torque or speed
    if match is None or not condition.get('phases'):
        return {}
    fixed, first, second = map(float, match.groups())
    values = np.full(len(times), np.nan)
    endpoints = {
        'linear_zero_to_steady_1': (0, first),
        'steady_1': (first, first),
        'linear_steady_1_to_steady_2': (first, second),
        'steady_2': (second, second),
        'linear_steady_2_to_steady_1': (second, first),
        'linear_steady_1_to_zero': (first, 0),
    }
    times = np.asarray(times, dtype=float)
    for phase in condition['phases']:
        if phase['phase'] not in endpoints:
            continue
        start, stop = phase['start_sec'], phase['stop_sec']
        if stop <= start:
            continue
        selected = (times >= start) & (times < stop)
        left, right = endpoints[phase['phase']]
        values[selected] = left + (right-left) * (times[selected]-start) / (stop-start)
    constant = np.full(len(times), fixed)
    return dict(nominal_speed_rpm=constant if torque else values,
                nominal_torque_nm=values if torque else constant)


def save_continual_report(run_dir, rows, elapsed, metadata, window, num_classes):
    """Every plotted point uses only predictions at or before that sample."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    n = len(rows)
    x = np.arange(1, n + 1)
    labels = np.array([r['target'] for r in rows], dtype=int)
    predictions = np.array([r['online_prediction'] for r in rows], dtype=int)
    curves = {}
    for stage in ('online', 'source', 'post_adapt'):
        correct = np.array([r[f'{stage}_correct'] for r in rows])
        prefix = np.r_[0, np.cumsum(correct)]
        starts = np.maximum(0, x - window)
        curves[f'{stage}_rolling_accuracy'] = 100 * (prefix[x] - prefix[starts]) / (x - starts)
        curves[f'{stage}_cumulative_accuracy'] = 100 * prefix[1:] / x
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    balanced, f1 = [], []
    for i, (target, pred) in enumerate(zip(labels, predictions)):
        cm[target, pred] += 1
        if i >= window:
            cm[labels[i-window], predictions[i-window]] -= 1
        support = cm.sum(axis=1)
        active = support > 0
        balanced.append(100 * np.mean(cm.diagonal()[active] / support[active]))
        denom = support + cm.sum(axis=0)
        valid = active
        f1.append(100 * np.mean(2 * cm.diagonal()[valid] / denom[valid]))
    curves['online_rolling_balanced_accuracy'] = balanced
    curves['online_rolling_macro_f1'] = f1
    condition = metadata.get('operating_condition', {})
    condition_curves = operating_condition_values(
        np.asarray(elapsed) + metadata['time_origin_sec'], condition)
    curves.update(condition_curves)
    for i, row in enumerate(rows):
        row.update(stream_index=i + 1, stream_time_sec=float(elapsed[i]),
                   window_start_sec=float(elapsed[i] + metadata['time_origin_sec']))
    with (run_dir / 'continual_curve.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['stream_index', 'stream_time_sec', 'sample_id', 'target'] + list(curves))
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({**{key: row[key] for key in ('stream_index', 'stream_time_sec', 'sample_id', 'target')},
                             **{key: float(value[i]) for key, value in curves.items()}})
    plot_time = np.asarray(elapsed) + metadata['time_origin_sec']
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    for stage, label in [('online', 'Online (before update)'), ('source', 'Frozen source'), ('post_adapt', 'After update (diagnostic)')]:
        axes[0].plot(plot_time, curves[f'{stage}_rolling_accuracy'], label=label, linewidth=1)
        axes[1].plot(plot_time, curves[f'{stage}_cumulative_accuracy'], label=label, linewidth=1)
    axes[0].plot(plot_time, balanced, label='Online balanced accuracy', alpha=.7, linewidth=1)
    axes[0].plot(plot_time, f1, label='Online macro F1', alpha=.7, linewidth=1)
    axes[0].set_ylabel(f'Trailing {window} samples (%)')
    axes[1].set_ylabel('Cumulative accuracy (%)')
    axes[2].scatter(plot_time, labels, s=3, label='True class')
    axes[2].scatter(plot_time, predictions, s=3, alpha=.35, label='Online prediction')
    axes[2].set_ylabel('Class')
    axes[2].set_yticks(range(num_classes))
    condition_ax = axes[3]
    if condition_curves:
        torque_ax = condition_ax.twinx()
        speed_line, = condition_ax.plot(plot_time, condition_curves['nominal_speed_rpm'],
                                       color='tab:blue', label='Nominal speed')
        torque_line, = torque_ax.plot(plot_time, condition_curves['nominal_torque_nm'],
                                     color='tab:orange', linestyle='--', label='Nominal torque')
        condition_ax.set_ylabel('Speed (rpm)', color='tab:blue')
        torque_ax.set_ylabel('Torque (Nm)', color='tab:orange')
        # A common zero baseline keeps constant conditions easy to read.
        for ax, values in [(condition_ax, condition_curves['nominal_speed_rpm']),
                           (torque_ax, condition_curves['nominal_torque_nm'])]:
            finite = values[np.isfinite(values)]
            if len(finite):
                ax.set_ylim(0, max(float(finite.max()) * 1.2, 1))
        condition_ax.legend(handles=[speed_line, torque_line], loc='lower left', fontsize=8)
        condition_ax.set_title('Operating condition: nominal profile at window start (not measured)', fontsize=10)
    else:
        condition_ax.plot(plot_time, np.full(n, condition.get('domain', rows[0].get('domain', 0))),
                          label='Domain ID (physical profile unavailable)')
        condition_ax.set_ylabel('Domain')
        condition_ax.set_title('Operating condition: domain ID only', fontsize=10)
    condition_ax.set_xlabel('Original recording time (s; window start)')
    event_groups = {}
    for event in metadata['events']:
        if event['class_id']:
            event_groups.setdefault(event['first_sample_time_sec'], []).append(event)
    for ax in axes:
        for actual_time, events in sorted(event_groups.items()):
            at = actual_time + metadata['time_origin_sec']
            ax.axvline(at, color='gray', linestyle='--', alpha=.5)
            if ax is axes[0]:
                classes = ','.join(f"c{event['class_id']}" for event in events)
                ax.text(at, 102, f"{classes} @ {at:.2f}s", fontsize=8, rotation=30)
        if ax is not condition_ax or not condition_curves:
            ax.legend(loc='lower left', fontsize=8)
        ax.grid(alpha=.2)
    phases = condition.get('phases', [])
    if phases:
        boundaries = sorted({float(p[key]) for p in phases for key in ('start_sec', 'stop_sec')})
        # Keep the enclosing phase boundaries visible, including the 60 s end
        # even though the last complete window starts slightly earlier.
        left = max((t for t in boundaries if t <= plot_time.min()), default=float(plot_time.min()))
        right = min((t for t in boundaries if t >= plot_time.max()), default=float(plot_time.max()))
        ticks = [t for t in boundaries if left <= t <= right]
        condition_ax.set_xticks(ticks)
        condition_ax.set_xlim(left, right)
    axes[0].set_ylim(0, 115)
    axes[1].set_ylim(0, 105)
    fig.suptitle('Continual fault-onset simulation — causal online evaluation')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(run_dir / f'continual_curve.{suffix}', dpi=160)
    plt.close(fig)
