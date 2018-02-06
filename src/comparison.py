from dataloader import CustomDataloader

def get_comparison_dataloader(comparison_dataset=None, **options):
    if not comparison_dataset:
        return
    comparison_options = options.copy()
    comparison_options['dataset'] = comparison_dataset
    comparison_options['last_batch'] = True
    comparison_options['shuffle'] = False
    comparison_dataloader = CustomDataloader(**comparison_options)
    return comparison_dataloader
