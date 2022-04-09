from sn6.cfg import tr_cfg
from sn6.model import BFE


def run():
    """Builds model, loads data, trains and evaluates"""
    print('initializing..')
    model = BFE(tr_cfg)
    print('loading data..')
    model.load_data()
    model.dataloader.preview_train_ds(n_show=4, n_rep=1, min_view=True)
    print('creating model..')
    model.load_model()
    print('starting training..')
    model.train(verbose=1)
    print('eval results:')
    model.view_results(n_show=6)

if __name__ == '__main__':
    run()