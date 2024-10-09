import datetime
import json
import os
import shutil
import matplotlib.pyplot as plt


class ExperimentTracker:
    def __init__(self, game_mode=False, base_path='trl_results'):
        self.epoch_data = {}
        self.epoch_evaluate_results = []
        self.game_mode = game_mode
        if game_mode: self.epoch_game_rewards = []
        self.exp_dir = self.get_experiment_dir(base_path)
        self.stats_path = os.path.join(self.exp_dir, 'exp.json')
        self.eval_path = os.path.join(self.exp_dir, 'evaluate_results.json')
    
    def get_experiment_dir(self, base_path):
        """Create and return a new experiment directory with a timestamp."""
        os.makedirs(base_path, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.game_mode:
            exp_dir_name = f'GAME_PLAY_EXP_{timestamp}'
        else:
            exp_dir_name = f'EXP_{timestamp}'
        exp_dir = os.path.join(base_path, exp_dir_name)
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir
    
    def record_experiment_details(self, script_path):
        """
        Back up the code of the experiment, including the main script,
        as well as the content in the utils/ and eval/ directories.
        """
        backup_dir = os.path.join(self.exp_dir, 'backup')
        os.makedirs(backup_dir, exist_ok=True)

        # Backup the main script
        original_filename = os.path.basename(script_path)
        script_backup_path = os.path.join(backup_dir, original_filename)
        shutil.copyfile(script_path, script_backup_path)

        # Backup utils folder
        base_script_dir = os.path.dirname(script_path)
        utils_path = os.path.join(base_script_dir, 'utils')
        if os.path.exists(utils_path) and os.path.isdir(utils_path):
            utils_backup_path = os.path.join(backup_dir, 'utils')
            shutil.copytree(utils_path, utils_backup_path, dirs_exist_ok=True)

        # Backup eval folder
        eval_path = os.path.join(base_script_dir, 'eval')
        if os.path.exists(eval_path) and os.path.isdir(eval_path):
            eval_backup_path = os.path.join(backup_dir, 'eval')
            shutil.copytree(eval_path, eval_backup_path, dirs_exist_ok=True)
    

    def record_epoch(self, epoch, **metrics):
        """Record any given metrics for the epoch."""
        for name, data in metrics.items():
            if name not in self.epoch_data:
                self.epoch_data[name] = []
            average = sum(data) / len(data) if len(data) > 0 else 0
            self.epoch_data[name].append((epoch, average))

        self._save_metrics()
        self._plot_metrics()

    def save_model(self, model, model_path='gpt2-alpaca-model'):
        """Saves the model within the experiment directory."""
        full_model_path = os.path.join(self.exp_dir, model_path)
        model.save_pretrained(full_model_path, push_to_hub=False)

    # def record_evaluation(self, epoch, results):
    #     # append results along with epoch to the list
    #     self.epoch_evaluate_results.append({'epoch': epoch, 'results': results})

    #     # call the save function
    #     self._save_evaluate_metrics()
    #     self._plot_eval_metrics()
    
    def record_evaluation(self, epoch, **metrics):
        """Record any given evaluation metrics for the epoch."""
        eval_data = {'epoch': epoch}
        for name, data in metrics.items():
            eval_data[name] = data
        self.epoch_evaluate_results.append(eval_data)

        self._save_evaluate_metrics()
        self._plot_eval_metrics()

    def _save_metrics(self):
        """Save the recorded metrics for all epochs to a JSON file."""
        stats = {metric: [] for metric in self.epoch_data}

        for metric, data in self.epoch_data.items():
            for epoch, value in data:
                stats[metric].append({'epoch': epoch, 'value': value})

        with open(self.stats_path, 'w') as stats_file:
            json.dump(stats, stats_file, indent=4)

    def _save_evaluate_metrics(self):
        """
        Save the evaluate results for each epoch into a JSON file.
        """

        # save the list with epoch and results to the json file
        with open(self.eval_path, 'w') as f:
            json.dump(self.epoch_evaluate_results, f, indent=4)  # Indent for readability
                

    def _plot_metrics(self):
        """Plot the learning curves for all recorded metrics."""
        for metric, data in self.epoch_data.items():
            epochs, values = zip(*data)
            self._plot_curve(epochs, values, f'Average {metric}', f'{metric}_curve.png')

    def _plot_curve(self, epochs, values, ylabel, filename):
        """Helper function to plot a curve and save it as an image."""
        plt.figure()
        plt.plot(epochs, values)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs Epoch')
        plt.savefig(os.path.join(self.exp_dir, filename))
        plt.close()

    
    def _plot_eval_metrics(self):
        """Plot all recorded evaluation metrics over the epochs."""
        # First, we extract all metric names (except for 'epoch') from the first record
        if not self.epoch_evaluate_results:
            return  # No data to plot

        metric_names = [metric for metric in self.epoch_evaluate_results[0] if metric != 'epoch']
        metrics_data = {metric: [] for metric in metric_names}
        epochs = []

        # Extract each metric data
        for evaluation in self.epoch_evaluate_results:
            epoch = evaluation['epoch']
            epochs.append(epoch)
            for metric in metric_names:
                metrics_data[metric].append(evaluation.get(metric, 0))

        # Plotting
        num_metrics = len(metric_names)
        plt.figure(figsize=(10, num_metrics * 4))  # Adjustable depending on number of metrics

        for i, (metric, values) in enumerate(metrics_data.items(), 1):
            plt.subplot(num_metrics, 1, i)
            plt.plot(epochs, values, label=metric)
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.title(f'{metric} Over Epochs')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, 'evaluation_metrics.png'))
        plt.close()

