import os
import yaml
import csv
from datetime import datetime
import torch
import json
import numpy as np


class ExperimentLogger:
    def __init__(self, save_dir="logs", phase=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if phase is None:
            self.exp_dir = os.path.join(save_dir, f"exp_{timestamp}")
        else:
            self.exp_dir = os.path.join(save_dir, phase, f"exp_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        self.metrics_file = os.path.join(self.exp_dir, "metrics.csv")
        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_ce_loss", "train_mask_loss", "train_rec_loss",
                "val_loss", "val_ce_loss", "val_mask_loss", "val_rec_loss",
            ])

        self.best_val_loss = float("inf")
        self.best_epoch = -1

    def log_epoch(self, epoch, loss, ce, mask, rec,
                  val_loss=None, val_ce=None, val_mask=None, val_rec=None):
        """
        Ghi log cho mot epoch. Cac tham so val_* la tuy chon (None neu khong
        chay validation trong epoch do) de tuong thich nguoc voi code cu.
        """
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, loss, ce, mask, rec,
                val_loss if val_loss is not None else "",
                val_ce if val_ce is not None else "",
                val_mask if val_mask is not None else "",
                val_rec if val_rec is not None else "",
            ])

    def is_best_val(self, val_loss, epoch):
        """Kiem tra xem epoch hien tai co dat val_loss thap nhat tu truoc den nay khong."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            return True
        return False

    def save_checkpoint(self, model, epoch, optimizer=None, scaler=None, suffix=None):
        """
        suffix: them hau to vao ten file, vi du 'best' de luu rieng best checkpoint
        (model_epoch_{epoch}_best.pth) ma khong bi ghi de boi checkpoint dinh ky.
        """
        if suffix:
            path = os.path.join(self.exp_dir, f"model_epoch_{epoch}_{suffix}.pth")
        else:
            path = os.path.join(self.exp_dir, f"model_epoch_{epoch}.pth")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        torch.save(checkpoint, path)
        return path

    def log_config(self, config):
        with open(os.path.join(self.exp_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f, sort_keys=False)

    def log_best_summary(self):
        """Ghi tom tat best epoch theo validation loss, goi sau khi train xong."""
        summary_path = os.path.join(self.exp_dir, "best_summary.txt")
        with open(summary_path, "w") as f:
            f.write("=" * 50 + "\n")
            f.write("DOMAIN ADAPTATION - BEST CHECKPOINT SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Best epoch: {self.best_epoch + 1}\n")
            f.write(f"Best validation loss: {self.best_val_loss:.6f}\n")
        return summary_path


class ExperimentClassificationLogger:
    def __init__(self, save_dir="logs", phase=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if phase is None:
            self.exp_dir = os.path.join(save_dir, f"exp_{timestamp}")
        else:
            self.exp_dir = os.path.join(save_dir, phase, f"exp_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        self.metrics_file = os.path.join(self.exp_dir, "metrics.csv")
        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "loss",
            ])

        self.cv_checkpoint_dir = os.path.join(self.exp_dir, "cv_checkpoints")
        os.makedirs(self.cv_checkpoint_dir, exist_ok=True)

    def log_epoch(self, epoch, loss):
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss])

    def save_checkpoint(self, model, epoch, optimizer=None, scaler=None):
        path = os.path.join(self.exp_dir, f"model_epoch_{epoch}.pth")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        torch.save(checkpoint, path)

    def log_config(self, config):
        with open(os.path.join(self.exp_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f, sort_keys=False)

    def log_metrics_cv(self, fold, epoch, train_loss, val_loss, val_accuracy):
        cv_metrics_file = os.path.join(self.exp_dir, f"cv_metrics_fold_{fold+1}.csv")

        if not os.path.exists(cv_metrics_file):
            with open(cv_metrics_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_accuracy"
                ])

        with open(cv_metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_accuracy])

        log_entry = f"Fold {fold+1}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}\n"
        with open(os.path.join(self.exp_dir, 'cv_training_log.txt'), 'a') as f:
            f.write(log_entry)

    def save_fold_checkpoint(self, model, fold, epoch, optimizer, scaler, accuracy):
        fold_dir = os.path.join(self.cv_checkpoint_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)

        checkpoint_path = os.path.join(fold_dir, f'best_model_fold_{fold+1}_epoch_{epoch+1}_acc_{accuracy:.4f}.pth')

        checkpoint = {
            'fold': fold + 1,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'accuracy': accuracy
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        best_info_file = os.path.join(fold_dir, 'best_fold_info.txt')
        with open(best_info_file, 'w') as f:
            f.write(f"Best model for Fold {fold+1}\n")
            f.write(f"Epoch: {epoch+1}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Checkpoint path: {checkpoint_path}\n")

    def log_cv_results(self, accuracies, losses):
        if isinstance(accuracies, list):
            accuracies = np.array(accuracies)
        if isinstance(losses, list):
            losses = np.array(losses)

        results = {
            'fold_accuracies': accuracies.tolist() if isinstance(accuracies, np.ndarray) else accuracies,
            'fold_losses': losses.tolist() if isinstance(losses, np.ndarray) else losses,
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'mean_loss': float(np.mean(losses)),
            'std_loss': float(np.std(losses))
        }

        results_path = os.path.join(self.exp_dir, 'cv_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        with open(os.path.join(self.exp_dir, 'cv_results.txt'), 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("CROSS-VALIDATION RESULTS\n")
            f.write("=" * 60 + "\n\n")

            for i, (acc, loss) in enumerate(zip(accuracies, losses)):
                f.write(f"Fold {i+1}:\n")
                f.write(f"  - Best Accuracy: {acc:.4f}\n")
                f.write(f"  - Best Loss: {loss:.4f}\n\n")

            f.write("-" * 60 + "\n")
            f.write(f"Mean Accuracy: {results['mean_accuracy']:.4f} +/- {results['std_accuracy']:.4f}\n")
            f.write(f"Mean Loss: {results['mean_loss']:.4f} +/- {results['std_loss']:.4f}\n")
            f.write("=" * 60 + "\n")

        print("\n" + "=" * 60)
        print("CROSS-VALIDATION RESULTS SAVED")
        print("=" * 60)
        print(f"Results saved to: {results_path}")
        print(f"Text summary saved to: {os.path.join(self.exp_dir, 'cv_results.txt')}")

    def log_best_fold_summary(self, best_fold_index, best_accuracy):
        summary_file = os.path.join(self.exp_dir, 'best_fold_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("BEST FOLD SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Best Fold: {best_fold_index + 1}\n")
            f.write(f"Best Accuracy: {best_accuracy:.4f}\n")
            f.write(f"Model checkpoint: {self.cv_checkpoint_dir}/fold_{best_fold_index + 1}/best_model_fold_{best_fold_index + 1}_*.pth\n")
            f.write("=" * 60 + "\n")

    def save_cv_training_curves(self, all_fold_histories):
        curves_data = {
            'fold_histories': all_fold_histories
        }

        curves_path = os.path.join(self.exp_dir, 'cv_training_curves.json')
        with open(curves_path, 'w') as f:
            json.dump(curves_data, f, indent=4)

        print(f"Training curves data saved to: {curves_path}")


# import os
# import yaml
# import csv
# from datetime import datetime
# import torch
# import json

# class ExperimentLogger:
#     def __init__(self, save_dir = "logs", phase = None):
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         if phase is None:
#             self.exp_dir = os.path.join(save_dir, f"exp_{timestamp}")
#         else:
#             self.exp_dir = os.path.join(save_dir, phase, f"exp_{timestamp}")
#         os.makedirs(self.exp_dir, exist_ok=True)

#         self.metrics_file = os.path.join(self.exp_dir, "metrics.csv")
#         with open(self.metrics_file, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 "epoch",
#                 "loss",
#                 "ce_loss",
#                 "mask_loss",
#                 "rec_loss"
#             ])

#     def log_epoch(self, epoch, loss, ce, mask, rec):
#         with open(self.metrics_file, "a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([epoch, loss, ce, mask, rec])
    
#     def save_checkpoint(self, model, epoch, optimizer=None, scaler=None):
#         path = os.path.join(self.exp_dir, f"model_epoch_{epoch}.pth")
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#         }
#         if optimizer is not None:
#             checkpoint['optimizer_state_dict'] = optimizer.state_dict()
#         if scaler is not None:
#             checkpoint['scaler_state_dict'] = scaler.state_dict()
#         torch.save(checkpoint, path)
    

#     def log_config(self, config):
#         with open(os.path.join(self.exp_dir, "config.yaml"), "w") as f:
#             yaml.dump(config, f, sort_keys=False)

# class ExperimentClassificationLogger:
#     def __init__(self, save_dir="logs", phase=None):
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         if phase is None:
#             self.exp_dir = os.path.join(save_dir, f"exp_{timestamp}")
#         else:
#             self.exp_dir = os.path.join(save_dir, phase, f"exp_{timestamp}")
#         os.makedirs(self.exp_dir, exist_ok=True)

#         self.metrics_file = os.path.join(self.exp_dir, "metrics.csv")
#         with open(self.metrics_file, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 "epoch",
#                 "loss",
#             ])
        
#         # Tạo thư mục cho checkpoints của cross-validation
#         self.cv_checkpoint_dir = os.path.join(self.exp_dir, "cv_checkpoints")
#         os.makedirs(self.cv_checkpoint_dir, exist_ok=True)

#     def log_epoch(self, epoch, loss):
#         with open(self.metrics_file, "a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([epoch, loss])
    
#     def save_checkpoint(self, model, epoch, optimizer=None, scaler=None):
#         path = os.path.join(self.exp_dir, f"model_epoch_{epoch}.pth")
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#         }
#         if optimizer is not None:
#             checkpoint['optimizer_state_dict'] = optimizer.state_dict()
#         if scaler is not None:
#             checkpoint['scaler_state_dict'] = scaler.state_dict()
#         torch.save(checkpoint, path)
    
#     def log_config(self, config):
#         with open(os.path.join(self.exp_dir, "config.yaml"), "w") as f:
#             yaml.dump(config, f, sort_keys=False)
    
#     # ============ CÁC PHƯƠNG THỨC MỚI CHO CROSS-VALIDATION ============
    
#     def log_metrics_cv(self, fold, epoch, train_loss, val_loss, val_accuracy):
#         """Log metrics cho mỗi fold trong quá trình cross-validation"""
#         # Tạo file metrics riêng cho từng fold nếu chưa có
#         cv_metrics_file = os.path.join(self.exp_dir, f"cv_metrics_fold_{fold+1}.csv")
        
#         # Kiểm tra nếu file chưa tồn tại thì tạo mới với header
#         if not os.path.exists(cv_metrics_file):
#             with open(cv_metrics_file, "w", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([
#                     "epoch",
#                     "train_loss",
#                     "val_loss",
#                     "val_accuracy"
#                 ])
        
#         # Ghi metrics
#         with open(cv_metrics_file, "a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([epoch, train_loss, val_loss, val_accuracy])
        
#         # Cũng ghi vào file log tổng hợp
#         log_entry = f"Fold {fold+1}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}\n"
#         with open(os.path.join(self.exp_dir, 'cv_training_log.txt'), 'a') as f:
#             f.write(log_entry)

#     def save_fold_checkpoint(self, model, fold, epoch, optimizer, scaler, accuracy):
#         """Lưu checkpoint cho mỗi fold"""
#         fold_dir = os.path.join(self.cv_checkpoint_dir, f'fold_{fold+1}')
#         os.makedirs(fold_dir, exist_ok=True)
        
#         checkpoint_path = os.path.join(fold_dir, f'best_model_fold_{fold+1}_epoch_{epoch+1}_acc_{accuracy:.4f}.pth')
        
#         checkpoint = {
#             'fold': fold + 1,
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scaler_state_dict': scaler.state_dict(),
#             'accuracy': accuracy
#         }
        
#         torch.save(checkpoint, checkpoint_path)
#         print(f"Checkpoint saved: {checkpoint_path}")
        
#         # Lưu thông tin về best accuracy
#         best_info_file = os.path.join(fold_dir, 'best_fold_info.txt')
#         with open(best_info_file, 'w') as f:
#             f.write(f"Best model for Fold {fold+1}\n")
#             f.write(f"Epoch: {epoch+1}\n")
#             f.write(f"Accuracy: {accuracy:.4f}\n")
#             f.write(f"Checkpoint path: {checkpoint_path}\n")

#     def log_cv_results(self, accuracies, losses):
#         """Log kết quả cuối cùng của cross-validation"""
#         # Chuyển đổi sang numpy array nếu cần
#         if isinstance(accuracies, list):
#             accuracies = np.array(accuracies)
#         if isinstance(losses, list):
#             losses = np.array(losses)
        
#         results = {
#             'fold_accuracies': accuracies.tolist() if isinstance(accuracies, np.ndarray) else accuracies,
#             'fold_losses': losses.tolist() if isinstance(losses, np.ndarray) else losses,
#             'mean_accuracy': float(np.mean(accuracies)),
#             'std_accuracy': float(np.std(accuracies)),
#             'mean_loss': float(np.mean(losses)),
#             'std_loss': float(np.std(losses))
#         }
        
#         # Lưu kết quả dạng JSON
#         results_path = os.path.join(self.exp_dir, 'cv_results.json')
#         with open(results_path, 'w') as f:
#             json.dump(results, f, indent=4)
        
#         # Lưu kết quả dạng text
#         with open(os.path.join(self.exp_dir, 'cv_results.txt'), 'w') as f:
#             f.write("="*60 + "\n")
#             f.write("CROSS-VALIDATION RESULTS\n")
#             f.write("="*60 + "\n\n")
            
#             for i, (acc, loss) in enumerate(zip(accuracies, losses)):
#                 f.write(f"Fold {i+1}:\n")
#                 f.write(f"  - Best Accuracy: {acc:.4f}\n")
#                 f.write(f"  - Best Loss: {loss:.4f}\n\n")
            
#             f.write("-"*60 + "\n")
#             f.write(f"Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}\n")
#             f.write(f"Mean Loss: {results['mean_loss']:.4f} ± {results['std_loss']:.4f}\n")
#             f.write("="*60 + "\n")
        
#         # In ra console
#         print("\n" + "="*60)
#         print("CROSS-VALIDATION RESULTS SAVED")
#         print("="*60)
#         print(f"Results saved to: {results_path}")
#         print(f"Text summary saved to: {os.path.join(self.exp_dir, 'cv_results.txt')}")
    
#     def log_best_fold_summary(self, best_fold_index, best_accuracy):
#         """Log thông tin về fold tốt nhất"""
#         summary_file = os.path.join(self.exp_dir, 'best_fold_summary.txt')
#         with open(summary_file, 'w') as f:
#             f.write("="*60 + "\n")
#             f.write("BEST FOLD SUMMARY\n")
#             f.write("="*60 + "\n")
#             f.write(f"Best Fold: {best_fold_index + 1}\n")
#             f.write(f"Best Accuracy: {best_accuracy:.4f}\n")
#             f.write(f"Model checkpoint: {self.cv_checkpoint_dir}/fold_{best_fold_index + 1}/best_model_fold_{best_fold_index + 1}_*.pth\n")
#             f.write("="*60 + "\n")
    
#     def save_cv_training_curves(self, all_fold_histories):
#         """Lưu dữ liệu để vẽ curves training cho tất cả các fold"""
#         curves_data = {
#             'fold_histories': all_fold_histories
#         }
        
#         curves_path = os.path.join(self.exp_dir, 'cv_training_curves.json')
#         with open(curves_path, 'w') as f:
#             json.dump(curves_data, f, indent=4)
        
#         print(f"Training curves data saved to: {curves_path}")



# # class ExperimentClassificationLogger:
# #     def __init__(self, save_dir = "logs", phase = None):
# #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #         if phase is None:
# #             self.exp_dir = os.path.join(save_dir, f"exp_{timestamp}")
# #         else:
# #             self.exp_dir = os.path.join(save_dir, phase, f"exp_{timestamp}")
# #         os.makedirs(self.exp_dir, exist_ok=True)

# #         self.metrics_file = os.path.join(self.exp_dir, "metrics.csv")
# #         with open(self.metrics_file, "w", newline="") as f:
# #             writer = csv.writer(f)
# #             writer.writerow([
# #                 "epoch",
# #                 "loss",
# #             ])

# #     def log_epoch(self, epoch, loss):
# #         with open(self.metrics_file, "a", newline="") as f:
# #             writer = csv.writer(f)
# #             writer.writerow([epoch, loss])
    
# #     def save_checkpoint(self, model, epoch, optimizer=None, scaler=None):
# #         path = os.path.join(self.exp_dir, f"model_epoch_{epoch}.pth")
# #         checkpoint = {
# #             'epoch': epoch,
# #             'model_state_dict': model.state_dict(),
# #         }
# #         if optimizer is not None:
# #             checkpoint['optimizer_state_dict'] = optimizer.state_dict()
# #         if scaler is not None:
# #             checkpoint['scaler_state_dict'] = scaler.state_dict()
# #         torch.save(checkpoint, path)
    

# #     def log_config(self, config):
# #         with open(os.path.join(self.exp_dir, "config.yaml"), "w") as f:
# #             yaml.dump(config, f, sort_keys=False)