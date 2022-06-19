import time
import os

def save_classification_report(classification_report, confusion_matrix_text, 
                               confusion_matrix_plot, save_path=''):
    time = current_time()

    try:
        dir_path = os.path.join(save_path, '_'.join(['report', time]))
        os.mkdir(dir_path)
        classification_report_filename = '_'.join(['classification_report', time+'.txt'])
        classification_report_path = os.path.join(dir_path, classification_report_filename)
        with open(classification_report_path, 'w+') as report_output_file:
            report_output_file.write(classification_report)
        cf_matrix_text_filename = '_'.join(['confusion_matrix', time+'.txt'])
        cf_matrix_text_path = os.path.join(dir_path, cf_matrix_text_filename)
        with open(cf_matrix_text_path, 'w+') as cf_mat_output_file:
            cf_mat_output_file.write(str(confusion_matrix_text))
        cf_matrix_plot_filename = '_'.join(['confusion_matrix_plot', time+'.png'])
        cf_matrix_plot_path = os.path.join(dir_path, cf_matrix_plot_filename)
        confusion_matrix_plot.figure.savefig(cf_matrix_plot_path)
    except:
        print('Directory Already Exists!')


def current_time():
    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    return current_time

