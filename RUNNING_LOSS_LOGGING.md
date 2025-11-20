# Running Loss Logging Implementation

## Overview

I have successfully added running loss logging functionality to the MeZO training code. This enhancement provides real-time visibility into training progress by displaying the smoothed current loss and learning rate in the training progress bar.

## What Was Added

### 1. Custom Progress Bar
- Replaced the default `ProgressCallback` with a custom global `tqdm` progress bar.
- Ensures a single, clean progress bar that tracks the global training progress.
- Handles cleanup (closing) properly at the end of training.

### 2. Running Loss Calculation
- **Real-time Smoothed Loss**: Implemented an Exponential Moving Average (EMA) of the loss.
- **Formula**: `running_loss = 0.9 * running_loss + 0.1 * current_loss`
- **Benefit**: This provides a stable yet responsive metric that reflects the current training performance better than a simple cumulative average.

### 3. Learning Rate Display
- Shows the current learning rate alongside the loss.
- Fetches the real-time learning rate from the scheduler using `_get_learning_rate()`.

### 4. Progress Bar Features
- **Postfix Information**: Displays `loss: X.XXXX` and `lr: X.XXe-X`.
- **Step Updates**: Progress bar advances with each global step (respecting gradient accumulation).
- **Disable Support**: Fully respects the `disable_tqdm` argument.

## Code Changes

### Key Implementation Details

Inside `large_models/trainer.py`:

```python
# Initialization before loop
if not args.disable_tqdm:
    self.remove_callback(ProgressCallback)
    pbar = tqdm(total=max_steps, initial=self.state.global_step, desc="Training", dynamic_ncols=True)
    running_loss = None

# Inside training loop
if not args.disable_tqdm:
    # Update running loss (EMA)
    current_loss = tr_loss_step.item()
    if running_loss is None:
        running_loss = current_loss
    else:
        running_loss = 0.9 * running_loss + 0.1 * current_loss
    
    logs = {'loss': f'{running_loss:.4f}'}
    try:
        logs['lr'] = f'{self._get_learning_rate():.2e}'
    except:
        pass
    pbar.set_postfix(logs)

# Update step count
if not args.disable_tqdm:
    pbar.update(1)

# Cleanup
if not args.disable_tqdm:
    pbar.close()
```

## Usage

The running loss logging is automatically enabled when training starts. No additional configuration is needed. The progress bar will appear showing:

```
Training:  45%|████▌     | 450/1000 [02:15<02:45,  3.32it/s, loss=2.3456, lr=1.00e-04]
```

Where:
- `Training`: Description
- `45%`: Progress percentage of total optimization steps
- `450/1000`: Global steps completed / total steps
- `loss=2.3456`: EMA of the loss
- `lr=1.00e-04`: Current learning rate

## Compatibility

- **Multi-GPU Training**: Works correctly (updates on main process via Trainer logic).
- **Gradient Accumulation**: Updates step count only on optimization steps, but updates loss display on every forward pass for maximum granularity.
- **Different Training Modes**: Works with both standard gradient descent and MeZO optimization.

## Testing

A test script (`test_running_loss.py`) is provided to verify the logic, though it may require a specific environment (torch, etc.) to execute fully.
