strain_task_list = ["TA100", "TA100_S9", "TA102", "TA102_S9", "TA104", "TA104_S9", "TA1535", "TA1535_S9", "TA1537", "TA1537_S9", "TA1538", "TA1538_S9", "TA97", "TA97_S9", "TA98", "TA98_S9"];
val_dataset_type_list = ["train", "val"];
test_dataset_type_list = ["trainval", "test"];

whattodo.bounding_box = 1;
whattodo.bounding_box_pca = 1;
whattodo.convex_hull = 0;
whattodo.leverage = 1;
whattodo.dist_centroid = 1;
whattodo.dist_knn_fix = 1;
whattodo.dist_knn_var = 0;
whattodo.pot_fun = 0;

options.pret_type = 'auto';
options.distance = 'euclidean';
options.lev_thr = 2.5000;
options.knnfix_k = 5;
options.dist_pct = 95;

for strain = strain_task_list
    val_datasets = {};
    for dataset_type = val_dataset_type_list
        filename = sprintf("../gmtamesAD_%s_%s.csv", strain, dataset_type);
        file = readmatrix(filename);
        val_datasets{end + 1} = file;
    end
    val_ad = ad_model(val_datasets{1}, val_datasets{2}, options, whattodo);
    writecell(val_ad.resume_table, sprintf("../ad_results/%s_val.csv", strain))

    test_datasets = {};
    for dataset_type = test_dataset_type_list
        filename = sprintf("../gmtamesAD_%s_%s.csv", strain, dataset_type);
        file = readmatrix(filename);
        test_datasets{end + 1} = file;
    end
    test_ad = ad_model(test_datasets{1}, test_datasets{2}, options, whattodo);
    writecell(test_ad.resume_table, sprintf("../ad_results/%s_test.csv", strain))

end