"""
creating workspace

project name

    dataset

    Out

        Models

            model_name_Aug

                Layer

                    Heads

                        train.zip
                        val.zip
                        model.h5
                        best.h5
                        loss.txt
                        config.pkl


                    4+

                        train.zip
                        val.zip
                        model.h5
                        best.h5
                        loss.txt
                        config.pkl

                    All
                        ....

            model_name_No_Aug
                ..
                    ...


"""
import os


def make_dir_if_none(d):
    if not os.path.exists(d):
        os.mkdir(d)
    else:
        print(f'Dir {d} already exists skipping...')


def create_workspace(root_dir, model_name, default_training_layers=['heads', '4+', 'all']):

    out_dir = os.path.join(root_dir, "Out")
    model_dir = os.path.join(out_dir, "Models")

    model_name_dir = os.path.join(model_dir, model_name)

    DIRS = {
        'PROJECT': root_dir,
        'OUT': out_dir,
        'MODEL': model_dir,
        'MODEL_NAME': model_name_dir,
    }

    for layer in default_training_layers:
        DIRS[f'{model_name}_{layer}'] = os.path.join(model_name_dir, layer)

    for d in DIRS:
        make_dir_if_none(DIRS[d])

    return DIRS
