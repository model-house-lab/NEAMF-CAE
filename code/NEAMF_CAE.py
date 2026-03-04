import pandas as pd
from utils import *
from sklearn.utils.class_weight import compute_class_weight
from keras.models import load_model
from model_upload import *
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, roc_curve, auc, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

def code_upload():
    pos_csv = pd.read_csv(r'..\\data\\image_positive.csv')
    pos_path = np.array(sorted(pos_csv['File Path'].tolist(), key=sort_key_zidian_1)).reshape((-1, 2))

    pos_text = []
    for _, row in pos_csv.iterrows():
        # Extracted 23 features (a-w)
        features = [row[c] for c in 'abcdefghijklmnopqrstuvw']
        pos_text.append(features)
    pos_text = np.array(pos_text[::2])

    pos_math_csv = pd.read_csv(r'..\\data\\math_positive.csv')
    pos_math_path = np.array(sorted(pos_math_csv['File Path'].tolist(), key=sort_key_zidian_1)).reshape((-1, 2))

    # Negative samples
    neg_csv = pd.read_csv(r'..\\data\\image_negative.csv')
    neg_path = np.array(sorted(neg_csv['File Path'].tolist(), key=sort_key_zidian_2)).reshape((-1, 2))

    neg_text = []
    for _, row in neg_csv.iterrows():
        features = [row[c] for c in 'abcdefghijklmnopqrstuvw']
        neg_text.append(features)
    neg_text = np.array(neg_text[::2])

    neg_math_csv = pd.read_csv(r'..\\data\\math_negative.csv')
    neg_math_path = np.array(sorted(neg_math_csv['File Path'].tolist(), key=sort_key_zidian_1)).reshape((-1, 2))

    # Merge datasets
    train_data_paths = np.concatenate((pos_path, neg_path), axis=0)
    train_math_paths = np.concatenate((pos_math_path, neg_math_path), axis=0)
    train_text = np.concatenate([pos_text, neg_text])
    train_labels = np.concatenate((np.ones(len(pos_path)), np.zeros(len(neg_path))), axis=0)

    # Load and normalize images
    X_img = read_images_from_folders(train_data_paths) / 255.0
    X_math = read_images_from_folders(train_math_paths) / 255.0

    print(f"Dataset Summary:")
    print(f" - Image shape: {X_img.shape}, Math shape: {X_math.shape}, Text shape: {train_text.shape}")

    # Train-test split
    X_img_train, X_img_val, X_math_train, X_math_val, X_text_train, X_text_val, y_train, y_val = train_test_split(
        X_img, X_math, train_text, train_labels,
        test_size=0.2, random_state=40, stratify=train_labels
    )

    print(f"Train samples: {len(y_train)} (Pos: {sum(y_train == 1)}, Neg: {sum(y_train == 0)})")
    print(f"Val samples:   {len(y_val)} (Pos: {sum(y_val == 1)}, Neg: {sum(y_val == 0)})")

    # Handle class imbalance
    cls_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = {0: cls_weights[0], 1: cls_weights[1]}
    print(f"Calculated Class Weights: Negative={weight_dict[0]:.2f}, Positive={weight_dict[1]:.2f}")

    input_shape = (32, 64, 64, 2)

    input_text_shape = 23

    autoencoder = load_model(
        r"..\\model\\NEAMF_CAE.tf",
        custom_objects={
        'multi_scale_module': multi_scale_module,
        'SEBlock': SEBlock,
        'spatial_attention_3d': spatial_attention_3d,
        'focal_region_loss': focal_region_loss,
        'Transformer3DOptimized': Transformer3DOptimized}
    )
    autoencoder.summary()

    feature_model = NEAMF_CAE(input_shape, input_text_shape, batch_size=2)
    feature_model.summary()

    for i in range(2,3):
        feature_model.layers[2*i+4].set_weights(autoencoder.layers[i].get_weights())
        feature_model.layers[2*i+4].trainable = False
        feature_model.layers[2*i+6].set_weights(autoencoder.layers[i].get_weights())
        feature_model.layers[2*i+6].trainable = False

    for i in range(3,4):
        feature_model.layers[2*i+3].set_weights(autoencoder.layers[i].get_weights())
        feature_model.layers[2*i+3].trainable = False
        feature_model.layers[2*i+5].set_weights(autoencoder.layers[i].get_weights())
        feature_model.layers[2*i+5].trainable = False
    for i in range(7,8):
        feature_model.layers[2*i+4].set_weights(autoencoder.layers[i].get_weights())
        feature_model.layers[2*i+4].trainable = False
        feature_model.layers[2*i+6].set_weights(autoencoder.layers[i].get_weights())
        feature_model.layers[2*i+6].trainable = False

    for i in range(8,9):
        feature_model.layers[2*i+3].set_weights(autoencoder.layers[i].get_weights())
        feature_model.layers[2*i+3].trainable = False
        feature_model.layers[2*i+5].set_weights(autoencoder.layers[i].get_weights())
        feature_model.layers[2*i+5].trainable = False

    for i in range(7,8):
        feature_model.layers[2*i+4].set_weights(autoencoder.layers[i].get_weights())
        feature_model.layers[2*i+4].trainable = False
        feature_model.layers[2*i+6].set_weights(autoencoder.layers[i].get_weights())
        feature_model.layers[2*i+6].trainable = False

    for i in range(8,9):
        feature_model.layers[2*i+3].set_weights(autoencoder.layers[i].get_weights())
        feature_model.layers[2*i+3].trainable = False
        feature_model.layers[2*i+5].set_weights(autoencoder.layers[i].get_weights())
        feature_model.layers[2*i+5].trainable = False

    for i in range(12,13):
        feature_model.layers[2*i+4].set_weights(autoencoder.layers[i].get_weights())
        feature_model.layers[2*i+4].trainable = False
        feature_model.layers[2*i+6].set_weights(autoencoder.layers[i].get_weights())
        feature_model.layers[2*i+6].trainable = False

    for i in range(13,14):
        feature_model.layers[2*i+3].set_weights(autoencoder.layers[i].get_weights())
        feature_model.layers[2*i+3].trainable = False
        feature_model.layers[2*i+5].set_weights(autoencoder.layers[i].get_weights())
        feature_model.layers[2*i+5].trainable = False


    feature_model.summary(line_length=150,positions=[0.30,0.60,0.7,1.])

    total_params = feature_model.count_params()


    def compute_flops(model):
        input_signature = [tf.TensorSpec(shape=(1,) + inp.shape[1:], dtype=inp.dtype) for inp in model.inputs]

        @tf.function
        def forward_pass(*args):
            return model(list(args))

        concrete_func = forward_pass.get_concrete_function(*input_signature)
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts['output'] = 'none'

        flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                              run_meta=run_meta, cmd='op', options=opts)
        return flops.total_float_ops


    total_flops = compute_flops(feature_model)

    print("=" * 60)
    print(f"📊 Computational Complexity")
    print(f"Total Parameters: {total_params / 1e6:.2f} M ")
    print(f"Total FLOPs:      {total_flops / 1e9:.2f} GFLOPs ")
    print("=" * 60)

    feature_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )


    test_csv_data = pd.read_csv(r'..\\data\\test_all.csv')
    test_data_path = test_csv_data['File Path'].tolist()
    test_data_path = np.array(sorted(test_data_path, key=sort_key_zidian_1)).reshape((-1, 2))

    combined_test_arrays = []
    for index, row in test_csv_data.iterrows():
        six_columns = [row['a'], row['b'], row['c'], row['d'], row['e'], row['f'], row['g'], row['h'], row['i'],
                       row['j'], row['k'], row['l'], row['m'], row['n'], row['o'], row['p'], row['q'], row['r'],
                       row['s'], row['t'], row['u'], row['v'], row['w'], row['x']]
        combined_test_arrays.append(six_columns)
    test_text_raw = combined_test_arrays[::2]

    test_csv_math = pd.read_csv(r'..\\data\\test_all_math.csv')
    test_math_path = test_csv_math['File Path'].tolist()
    test_math_path = np.array(sorted(test_math_path, key=sort_key_zidian_1)).reshape((-1, 2))

    X_test_img = read_images_from_folders(test_data_path) / 255.0
    X_test_math = read_images_from_folders(test_math_path) / 255.0

    test_text_array = np.array(test_text_raw, dtype=np.float32)
    X_test_text = test_text_array[:, :-1]
    y_test = test_text_array[:, -1].astype(int)



    initial_weights_path = 'initial_surrogate_weights.h5'
    feature_model.save_weights(initial_weights_path)

    num_runs = 5
    val_results = {k: [] for k in ['auc', 'acc', 'prec', 'rec', 'spec', 'npv', 'f1', 'bacc',
                                   'ci_auc', 'ci_acc', 'ci_prec', 'ci_rec', 'ci_spec', 'ci_npv', 'ci_f1', 'ci_bacc']}
    test_results = {k: [] for k in ['auc', 'acc', 'prec', 'rec', 'spec', 'npv', 'f1', 'bacc',
                                    'ci_auc', 'ci_acc', 'ci_prec', 'ci_rec', 'ci_spec', 'ci_npv', 'ci_f1', 'ci_bacc']}
    ensemble_test_probs = []

    for run_id in range(num_runs):
        print(f"             start {run_id + 1}  experiment            ")

        feature_model.load_weights(initial_weights_path)
        run_weight_path = f'..\\model\\best_weights_run_{run_id + 1}.h5'
        checkpoint = ModelCheckpoint(filepath=run_weight_path, monitor='val_accuracy', mode='max', save_best_only=True,
                                     save_weights_only=True, verbose=0)

        feature_model.fit(
            x=[X_img_train, X_math_train, X_text_train], y=y_train,
            validation_data=([X_img_val, X_math_val, X_text_val], y_val),
            batch_size=2, epochs=30, class_weight=weight_dict, callbacks=[checkpoint], shuffle=True, verbose=1
        )

        if os.path.exists(run_weight_path):
            feature_model.load_weights(run_weight_path)

        val_prob = feature_model.predict([X_img_val, X_math_val, X_text_val], batch_size=2)
        val_class = (val_prob > 0.5).astype(int)


        v_auc = roc_auc_score(y_val, val_prob)
        v_acc = accuracy_score(y_val, val_class)
        v_prec = precision_score(y_val, val_class, zero_division=0)
        v_rec = recall_score(y_val, val_class, zero_division=0)
        v_f1 = f1_score(y_val, val_class, zero_division=0)
        v_bacc = balanced_accuracy_score(y_val, val_class)
        v_cm = confusion_matrix(y_val, val_class, labels=[0, 1])
        v_tn, v_fp, v_fn, v_tp = v_cm.ravel()
        v_spec = v_tn / (v_tn + v_fp) if (v_tn + v_fp) > 0 else 0.0
        v_npv = v_tn / (v_tn + v_fn) if (v_tn + v_fn) > 0 else 0.0

        v_ci_auc, v_ci_acc, v_ci_prec, v_ci_rec, v_ci_spec, v_ci_npv, v_ci_f1, v_ci_bacc = compute_95ci_bootstrap(y_val,
                                                                                                                  val_prob,
                                                                                                                  val_class)

        for k, v in zip(['auc', 'acc', 'prec', 'rec', 'spec', 'npv', 'f1', 'bacc'],
                        [v_auc, v_acc, v_prec, v_rec, v_spec, v_npv, v_f1, v_bacc]):
            val_results[k].append(v)
        for k, v in zip(['ci_auc', 'ci_acc', 'ci_prec', 'ci_rec', 'ci_spec', 'ci_npv', 'ci_f1', 'ci_bacc'],
                        [v_ci_auc, v_ci_acc, v_ci_prec, v_ci_rec, v_ci_spec, v_ci_npv, v_ci_f1, v_ci_bacc]):
            val_results[k].append(v)

        test_prob = feature_model.predict([X_test_img, X_test_math, X_test_text], batch_size=2)
        ensemble_test_probs.append(test_prob)
        test_class = (test_prob > 0.5).astype(int)

        t_auc = roc_auc_score(y_test, test_prob)
        t_acc = accuracy_score(y_test, test_class)
        t_prec = precision_score(y_test, test_class, zero_division=0)
        t_rec = recall_score(y_test, test_class, zero_division=0)
        t_f1 = f1_score(y_test, test_class, zero_division=0)
        t_bacc = balanced_accuracy_score(y_test, test_class)
        t_cm = confusion_matrix(y_test, test_class, labels=[0, 1])
        t_tn, t_fp, t_fn, t_tp = t_cm.ravel()
        t_spec = t_tn / (t_tn + t_fp) if (t_tn + t_fp) > 0 else 0.0
        t_npv = t_tn / (t_tn + t_fn) if (t_tn + t_fn) > 0 else 0.0

        t_ci_auc, t_ci_acc, t_ci_prec, t_ci_rec, t_ci_spec, t_ci_npv, t_ci_f1, t_ci_bacc = compute_95ci_bootstrap(y_test,
                                                                                                                  test_prob,
                                                                                                                  test_class)

        for k, v in zip(['auc', 'acc', 'prec', 'rec', 'spec', 'npv', 'f1', 'bacc'],
                        [t_auc, t_acc, t_prec, t_rec, t_spec, t_npv, t_f1, t_bacc]):
            test_results[k].append(v)
        for k, v in zip(['ci_auc', 'ci_acc', 'ci_prec', 'ci_rec', 'ci_spec', 'ci_npv', 'ci_f1', 'ci_bacc'],
                        [t_ci_auc, t_ci_acc, t_ci_prec, t_ci_rec, t_ci_spec, t_ci_npv, t_ci_f1, t_ci_bacc]):
            test_results[k].append(v)

        roc_save_dir = r"..\\weight"
        os.makedirs(roc_save_dir, exist_ok=True)
        np.savez(os.path.join(roc_save_dir, f"CAE_TestSet_Run{run_id + 1}.npz"), fpr=roc_curve(y_test, test_prob)[0],
                 tpr=roc_curve(y_test, test_prob)[1], auc=t_auc)

    print("\n" + "=" * 70)
    print("   📊 最终实验平均表现 -> Mean ± Std (95% CI: Lower - Upper)")
    print("=" * 70)


    def print_metrics(results_dict, dataset_name):
        print(f"[{dataset_name}]")
        metrics_list = [('AUC', 'auc', 'ci_auc'), ('Accuracy', 'acc', 'ci_acc'),
                        ('Precision', 'prec', 'ci_prec'), ('Recall', 'rec', 'ci_rec'),
                        ('Specificity', 'spec', 'ci_spec'), ('NPV', 'npv', 'ci_npv'),
                        ('F1-score', 'f1', 'ci_f1'), ('Balanced Acc', 'bacc', 'ci_bacc')]
        for name, m_key, ci_key in metrics_list:
            mean_val = np.mean(results_dict[m_key])
            std_val = np.std(results_dict[m_key])
            mean_ci = np.mean(results_dict[ci_key], axis=0)
            print(f"{name:13s}: {mean_val:.4f} ± {std_val:.4f}  (95% CI: {mean_ci[0]:.4f} - {mean_ci[1]:.4f})")
        print("-" * 50)


    print_metrics(val_results, "Validation Set")
    print_metrics(test_results, "Independent Test Set")


    final_ensemble_prob = np.mean(ensemble_test_probs, axis=0)
    final_ensemble_class = (final_ensemble_prob > 0.5).astype(int)

    fpr_ens, tpr_ens, _ = roc_curve(y_test, final_ensemble_prob)
    e_auc = roc_auc_score(y_test, final_ensemble_prob)
    np.savez(os.path.join(roc_save_dir, "NEAMF_CAE_TestSet_Ensemble_ROC.npz"),
             fpr=roc_curve(y_test, final_ensemble_prob)[0],
             tpr=roc_curve(y_test, final_ensemble_prob)[1],
             auc=e_auc,
             y_true=y_test,
             y_prob=final_ensemble_prob
             )

code_upload()