%% run_casia_full_eval_matlab.m
% Pełna ewaluacja klasycznej metody MATLAB Iris-Recognition / Masek-Kovesi dla CASIA-IrisV1.
%
% Co zapisuje:
% 1) czasy etapów segmentacji: źrenica, tęczówka, powieki, rzęsy,
% 2) czasy etapów procesu: normalizacja, kodowanie, matching,
% 3) maski predykcji źrenicy utworzone z circlepupil,
% 4) IoU, Dice, pixel accuracy, precision, recall, specificity względem binarnych masek GT,
% 5) nakładki błędnych segmentacji,
% 6) wyniki identyfikacji top-1 dystansem Hamminga.
%
% Ważne: skrypt zakłada oryginalne funkcje w folderze matlab/fnc projektu Iris-Recognition-master.
% Uruchom w MATLAB-ie z poziomu dowolnego folderu albo ustaw poniżej MATLAB_CODE_DIR.

clear; clc;

%% ========================================================================
%  PRZENOŚNE ŚCIEŻKI DOMYŚLNE
%  ========================================================================
%
% Struktura repozytorium:
%   src/matlab/                         - ten skrypt (kod ewaluacyjny projektu)
%   third_party/Iris-Recognition-master/matlab/fnc/
%                                      - zewnętrzna implementacja metody
%   data/CASIA1_BMP/                   - obrazy CASIA-IrisV1 (nie są dystrybuowane)
%   data/CASIA1_JPG/                   - opcjonalna konwersja JPG
%   data/ground_truth/pupil_masks/     - binarne maski źrenicy
%   results/matlab/                    - wyniki

SCRIPT_DIR = fileparts(mfilename('fullpath'));
REPO_ROOT = fileparts(fileparts(SCRIPT_DIR));

MATLAB_CODE_DIR = fullfile(REPO_ROOT, 'third_party', 'Iris-Recognition-master', 'matlab');
GT_DIR = fullfile(REPO_ROOT, 'data', 'ground_truth', 'pupil_masks');
OUTPUT_DIR = fullfile(REPO_ROOT, 'results', 'matlab');

DATASETS = struct([]);
DATASETS(1).name = 'CASIA1_JPG';
DATASETS(1).images_dir = fullfile(REPO_ROOT, 'data', 'CASIA1_JPG');
DATASETS(1).extensions = {'.jpg', '.jpeg', '.png', '.bmp'};

DATASETS(2).name = 'CASIA1_BMP';
DATASETS(2).images_dir = fullfile(REPO_ROOT, 'data', 'CASIA1_BMP');
DATASETS(2).extensions = {'.bmp', '.jpg', '.jpeg', '.png'};

EYELASHES_THRESHOLD = 80;
RADIAL_RES = 20;
ANGULAR_RES = 240;
NSCALES = 1;
MIN_WAVELENGTH = 18;
MULT = 1;
SIGMA_ONF = 0.5;
BAD_IOU_THRESHOLD = 0.50;
BAD_DICE_THRESHOLD = 0.65;
N_WORST_EXAMPLES = 80;

if exist(fullfile(MATLAB_CODE_DIR, 'fnc'), 'dir') ~= 7
    error('Nie znaleziono folderu fnc: %s', fullfile(MATLAB_CODE_DIR, 'fnc'));
end
addpath(fullfile(MATLAB_CODE_DIR, 'fnc'));

if exist(OUTPUT_DIR, 'dir') ~= 7
    mkdir(OUTPUT_DIR);
end

fprintf('\n==============================================================\n');
fprintf('START EWALUACJI MATLAB — CASIA V1, MASEK/KOVESI\n');
fprintf('Kod MATLAB: %s\n', MATLAB_CODE_DIR);
fprintf('GT: %s\n', GT_DIR);
fprintf('Wyniki: %s\n', OUTPUT_DIR);
fprintf('==============================================================\n');

gtMap = buildGtMap(GT_DIR);
allSummary = struct();
allTimer = tic;

for d = 1:numel(DATASETS)
    ds = DATASETS(d);
    summary = evaluateOneDatasetMatlab(ds, gtMap, GT_DIR, OUTPUT_DIR, ...
        EYELASHES_THRESHOLD, RADIAL_RES, ANGULAR_RES, NSCALES, ...
        MIN_WAVELENGTH, MULT, SIGMA_ONF, BAD_IOU_THRESHOLD, BAD_DICE_THRESHOLD, N_WORST_EXAMPLES);
    allSummary.(ds.name) = summary;
end

allSummary.total_runtime_s = toc(allTimer);
save(fullfile(OUTPUT_DIR, 'summary_all_datasets_matlab.mat'), 'allSummary');

fprintf('\n==============================================================\n');
fprintf('ZAKOŃCZONO CAŁY EKSPERYMENT MATLAB\n');
fprintf('Folder wyników: %s\n', OUTPUT_DIR);
fprintf('Całkowity czas: %.3f s\n', allSummary.total_runtime_s);
fprintf('==============================================================\n');


%% ========================================================================
%  FUNKCJE LOKALNE
%  ========================================================================

function summary = evaluateOneDatasetMatlab(ds, gtMap, gtDir, outputDir, eyelashesThreshold, radialRes, angularRes, nscales, minWaveLength, mult, sigmaOnf, badIouThreshold, badDiceThreshold, nWorstExamples)
    datasetOut = fullfile(outputDir, ds.name);
    overlaysBadDir = fullfile(datasetOut, 'bad_segmentations_overlays');
    overlaysMissingGtDir = fullfile(datasetOut, 'missing_gt_overlays');
    overlaysFailedDir = fullfile(datasetOut, 'failed_processing');
    masksPredDir = fullfile(datasetOut, 'predicted_pupil_masks');
    worstDir = fullfile(datasetOut, 'worst_examples');
    ensureDir(datasetOut); ensureDir(overlaysBadDir); ensureDir(overlaysMissingGtDir); ensureDir(overlaysFailedDir); ensureDir(masksPredDir); ensureDir(worstDir);

    imageFiles = findImagesRecursive(ds.images_dir, ds.extensions);
    nImages = numel(imageFiles);

    fprintf('\n==============================================================\n');
    fprintf('[DATASET] %s\n', ds.name);
    fprintf('[OBRAZY]  %s\n', ds.images_dir);
    fprintf('[GT]      %s\n', gtDir);
    fprintf('[WYNIKI]  %s\n', datasetOut);
    fprintf('[LICZBA OBRAZÓW] %d\n', nImages);
    fprintf('[LICZBA MASEK GT W INDEKSIE] %d\n', gtMap.Count);
    fprintf('==============================================================\n');

    rows = struct([]);
    featureRecords = struct([]);
    featCount = 0;

    globalTP = 0; globalFP = 0; globalFN = 0; globalTN = 0;

    for i = 1:nImages
        imgPath = imageFiles{i};
        relPath = relativePath(imgPath, ds.images_dir);
        [~, stem, ext] = fileparts(imgPath);
        identity = identityFromFilename(stem, imgPath);

        if i == 1 || mod(i,25) == 0 || i == nImages
            fprintf('[%s] %d/%d: %s\n', ds.name, i, nImages, relPath);
        end

        row = emptyImageRow();
        row.dataset = string(ds.name);
        row.index = i;
        row.image_path = string(imgPath);
        row.rel_path = string(relPath);
        row.filename = string([stem ext]);
        row.stem = string(stem);
        row.identity = string(identity);
        row.status = string('started');
        row.error = string('');
        row.gt_path = string('');
        row.gt_note = string('');
        row.bad_segmentation = string('');

        try
            result = detailedExtractFeatureMatlab(imgPath, eyelashesThreshold, radialRes, angularRes, nscales, minWaveLength, mult, sigmaOnf);
            eyeimage = result.image;
            circlepupil = result.circlepupil;
            circleiris = result.circleiris;

            row.status = string('ok');
            row.read_s = result.times.read_s;
            row.inner_pupil_s = result.times.inner_pupil_s;
            row.outer_iris_s = result.times.outer_iris_s;
            row.top_eyelid_s = result.times.top_eyelid_s;
            row.bottom_eyelid_s = result.times.bottom_eyelid_s;
            row.eyelashes_s = result.times.eyelashes_s;
            row.segmentation_total_s = result.times.segmentation_total_s;
            row.normalization_s = result.times.normalization_s;
            row.encoding_s = result.times.encoding_s;
            row.feature_total_s = result.times.feature_total_s;

            row.pupil_y = circlepupil(1); row.pupil_x = circlepupil(2); row.pupil_r = circlepupil(3);
            row.iris_y = circleiris(1); row.iris_x = circleiris(2); row.iris_r = circleiris(3);

            tMask = tic;
            predPupil = circleMask(size(eyeimage), circlepupil);
            predIris = circleMask(size(eyeimage), circleiris) & ~predPupil;
            row.pupil_mask_s = toc(tMask);
            row.pred_pupil_area_px = nnz(predPupil);
            row.pred_iris_annulus_area_px = nnz(predIris);

            predMaskPath = fullfile(masksPredDir, [stem '_pred_pupil.png']);
            imwrite(uint8(predPupil) * 255, predMaskPath);
            row.pred_pupil_mask_path = string(predMaskPath);

            gtPath = locateGtMaskMatlab(stem, gtMap);
            gtPupil = [];
            if strlength(gtPath) > 0
                row.gt_path = string(gtPath);
                tGt = tic;
                [gtPupil, gtNote] = loadGtMaskMatlab(char(gtPath), size(eyeimage));
                row.gt_load_s = toc(tGt);
                row.gt_note = string(gtNote);
                row.gt_area_px = nnz(gtPupil);

                tMetrics = tic;
                met = binaryMetricsMatlab(predPupil, gtPupil);
                row.tp = met.tp; row.fp = met.fp; row.fn = met.fn; row.tn = met.tn;
                row.iou = met.iou; row.dice = met.dice; row.pixel_accuracy = met.pixel_accuracy;
                row.precision = met.precision; row.recall = met.recall; row.specificity = met.specificity;
                row.fpr = met.fpr; row.fnr = met.fnr;
                row.metrics_s = toc(tMetrics);

                globalTP = globalTP + met.tp;
                globalFP = globalFP + met.fp;
                globalFN = globalFN + met.fn;
                globalTN = globalTN + met.tn;

                isBad = (met.iou < badIouThreshold) || (met.dice < badDiceThreshold);
                row.bad_segmentation = string(num2str(isBad));

                if isBad
                    diag = makeDiagnosticImageMatlab(eyeimage, predPupil, gtPupil, circlepupil, circleiris, {
                        ['file: ' relPath], ...
                        sprintf('IoU=%.4f, Dice=%.4f, PA=%.4f', met.iou, met.dice, met.pixel_accuracy), ...
                        sprintf('pupil pred y/x/r = %.0f/%.0f/%.0f', circlepupil(1), circlepupil(2), circlepupil(3)), ...
                        sprintf('iris y/x/r = %.0f/%.0f/%.0f', circleiris(1), circleiris(2), circleiris(3)), ...
                        ['GT: ' char(gtPath)]});
                    outName = sprintf('%04d_%s_IoU_%.3f_Dice_%.3f.png', i, stem, met.iou, met.dice);
                    imwrite(diag, fullfile(overlaysBadDir, outName));
                end
            else
                row.status = string('ok_missing_gt');
                row.bad_segmentation = string('missing_gt');
                diag = makeDiagnosticImageMatlab(eyeimage, predPupil, [], circlepupil, circleiris, {
                    ['file: ' relPath], 'Brak maski GT dla tego obrazu.', ...
                    sprintf('pupil pred y/x/r = %.0f/%.0f/%.0f', circlepupil(1), circlepupil(2), circlepupil(3)), ...
                    sprintf('iris y/x/r = %.0f/%.0f/%.0f', circleiris(1), circleiris(2), circleiris(3))});
                imwrite(diag, fullfile(overlaysMissingGtDir, sprintf('%04d_%s_missing_gt.png', i, stem)));
            end

            featCount = featCount + 1;
            featureRecords(featCount).identity = identity;
            featureRecords(featCount).image_path = imgPath;
            featureRecords(featCount).rel_path = relPath;
            featureRecords(featCount).template = result.template;
            featureRecords(featCount).mask = result.mask;
            featureRecords(featCount).feature_total_s = result.times.feature_total_s;
            featureRecords(featCount).segmentation_total_s = result.times.segmentation_total_s;

        catch ME
            row.status = string('failed');
            row.error = string(ME.message);
            try
                failedImage = imread(imgPath);
                if ndims(failedImage) == 3; failedImage = rgb2gray(failedImage); end
                diag = makeDiagnosticImageMatlab(failedImage, [], [], [], [], {['file: ' relPath], 'BŁĄD PRZETWARZANIA / SEGMENTACJI', ME.message});
                imwrite(diag, fullfile(overlaysFailedDir, sprintf('%04d_%s_FAILED.png', i, stem)));
            catch
            end
        end

        if isempty(rows)
            rows = row;
        else
            rows(end+1) = row; %#ok<AGROW>
        end
    end

    imageTable = struct2table(rows);
    writetable(imageTable, fullfile(datasetOut, 'per_image_segmentation_and_feature_times.csv'));

    idRows = evaluateIdentificationTop1Matlab(featureRecords, datasetOut);

    summary = buildSummaryMatlab(ds.name, rows, idRows, globalTP, globalFP, globalFN, globalTN, badIouThreshold, badDiceThreshold);
    save(fullfile(datasetOut, 'summary.mat'), 'summary');
    writeSummaryTxtMatlab(fullfile(datasetOut, 'summary.txt'), summary);
    makePlotsMatlab(datasetOut, rows);

    % Kopiowanie najgorszych przykładów do osobnego folderu.
    try
        iouVals = [rows.iou];
        valid = ~isnan(iouVals);
        validIdx = find(valid);
        [~, ord] = sort(iouVals(valid));
        take = min(nWorstExamples, numel(ord));
        for k = 1:take
            rowIdx = validIdx(ord(k));
            stemK = char(rows(rowIdx).stem);
            files = dir(fullfile(overlaysBadDir, ['*_' stemK '_IoU_*.png']));
            if ~isempty(files)
                copyfile(fullfile(files(1).folder, files(1).name), fullfile(worstDir, sprintf('rank_%03d_%s', k, files(1).name)));
            end
        end
    catch
    end

    fprintf('\n[%s] ZAKOŃCZONO. Wyniki: %s\n', ds.name, datasetOut);
    fprintf('[%s] IoU mean: %.6f | Dice mean: %.6f | Top-1: %.6f\n', ds.name, summary.pupil_iou_mean, summary.pupil_dice_mean, summary.identification_top1_accuracy);
end

function result = detailedExtractFeatureMatlab(imgPath, eyelashesThreshold, radialRes, angularRes, nscales, minWaveLength, mult, sigmaOnf)
    totalTimer = tic;
    t = tic;
    eyeimage = imread(imgPath);
    if ndims(eyeimage) == 3
        eyeimage = rgb2gray(eyeimage);
    end
    times.read_s = toc(t);

    segTimer = tic;
    t = tic;
    [rowp, colp, rp] = SearchInnerBoundary(double(eyeimage));
    times.inner_pupil_s = toc(t);

    t = tic;
    [row, col, r] = SearchOuterBoundary(double(eyeimage), rowp, colp, rp);
    times.outer_iris_s = toc(t);

    rowp = round(rowp); colp = round(colp); rp = round(rp);
    row = round(row); col = round(col); r = round(r);
    circleiris = [row, col, r];

    rowd = double(row); cold = double(col); rd = double(r);
    irl = round(rowd-rd); iru = round(rowd+rd); icl = round(cold-rd); icu = round(cold+rd);
    imgsize = size(eyeimage);
    if irl < 1; irl = 1; end
    if icl < 1; icl = 1; end
    if iru > imgsize(1); iru = imgsize(1); end
    if icu > imgsize(2); icu = imgsize(2); end

    imagepupil = double(eyeimage(irl:iru, icl:icu));
    rowpLocal = double(rowp) - irl;
    colpLocal = double(colp) - icl;
    rLocal = double(rp);
    rowLocal = double(irl) + rowpLocal;
    colLocal = double(icl) + colpLocal;
    circlepupil = [round(rowLocal), round(colLocal), rLocal];

    imagewithnoise = double(eyeimage);

    t = tic;
    try
        topEnd = floor(rowpLocal - rLocal);
        if topEnd >= 1
            topeyelid = uint8(imagepupil(1:topEnd, :));
            lines = findline(topeyelid);
            if size(lines,1) > 0
                [xl, yl] = linecoords(lines, size(topeyelid));
                yl = double(yl) + irl - 1;
                xl = double(xl) + icl - 1;
                yla = max(yl);
                y2 = 1:yla;
                valid = yl >= 1 & yl <= size(eyeimage,1) & xl >= 1 & xl <= size(eyeimage,2);
                ind3 = sub2ind(size(eyeimage), round(yl(valid)), round(xl(valid)));
                imagewithnoise(ind3) = NaN;
                xValid = round(xl(valid));
                xValid = xValid(xValid >= 1 & xValid <= size(eyeimage,2));
                y2 = round(y2); y2 = y2(y2 >= 1 & y2 <= size(eyeimage,1));
                imagewithnoise(y2, xValid) = NaN;
            end
        end
    catch
    end
    times.top_eyelid_s = toc(t);

    t = tic;
    try
        botStart = ceil(rowpLocal + rLocal);
        if botStart >= 1 && botStart <= size(imagepupil,1)
            bottomeyelid = uint8(imagepupil(botStart:size(imagepupil,1), :));
            lines = findline(bottomeyelid);
            if size(lines,1) > 0
                [xl, yl] = linecoords(lines, size(bottomeyelid));
                yl = double(yl) + irl + rowpLocal + rLocal - 2;
                xl = double(xl) + icl - 1;
                yla = min(yl);
                y2 = yla:size(eyeimage,1);
                valid = yl >= 1 & yl <= size(eyeimage,1) & xl >= 1 & xl <= size(eyeimage,2);
                ind4 = sub2ind(size(eyeimage), round(yl(valid)), round(xl(valid)));
                imagewithnoise(ind4) = NaN;
                xValid = round(xl(valid));
                xValid = xValid(xValid >= 1 & xValid <= size(eyeimage,2));
                y2 = round(y2); y2 = y2(y2 >= 1 & y2 <= size(eyeimage,1));
                imagewithnoise(y2, xValid) = NaN;
            end
        end
    catch
    end
    times.bottom_eyelid_s = toc(t);

    t = tic;
    ref = eyeimage < eyelashesThreshold;
    coords = find(ref == 1);
    imagewithnoise(coords) = NaN;
    times.eyelashes_s = toc(t);
    times.segmentation_total_s = toc(segTimer);

    t = tic;
    [polar_array, noise_array] = normaliseiris(imagewithnoise, ...
        circleiris(2), circleiris(1), circleiris(3), ...
        circlepupil(2), circlepupil(1), circlepupil(3), ...
        imgPath, radialRes, angularRes);
    times.normalization_s = toc(t);

    t = tic;
    [template, mask] = encode(polar_array, noise_array, nscales, minWaveLength, mult, sigmaOnf);
    times.encoding_s = toc(t);
    times.feature_total_s = toc(totalTimer);

    result.image = eyeimage;
    result.circleiris = circleiris;
    result.circlepupil = circlepupil;
    result.imagewithnoise = imagewithnoise;
    result.polar_array = polar_array;
    result.noise_array = noise_array;
    result.template = template;
    result.mask = mask;
    result.times = times;
end

function idRows = evaluateIdentificationTop1Matlab(featureRecords, datasetOut)
    idRows = struct([]);
    if isempty(featureRecords)
        fid = fopen(fullfile(datasetOut, 'identification_top1_results.csv'), 'w');
        fprintf(fid, 'no_features\n'); fclose(fid);
        return;
    end

    identities = unique(string({featureRecords.identity}));
    gallery = struct([]);
    probes = struct([]);
    gCount = 0; pCount = 0;
    for i = 1:numel(identities)
        ident = identities(i);
        idx = find(string({featureRecords.identity}) == ident);
        if isempty(idx); continue; end
        gCount = gCount + 1;
        gallery(gCount).identity = char(ident);
        gallery(gCount).record = featureRecords(idx(1));
        if numel(idx) > 1
            for j = 2:numel(idx)
                pCount = pCount + 1;
                probes(pCount).record = featureRecords(idx(j));
            end
        end
    end

    fprintf('[IDENTYFIKACJA MATLAB] gallery identities: %d, probes: %d\n', numel(gallery), numel(probes));

    for p = 1:numel(probes)
        probe = probes(p).record;
        tTotal = tic;
        bestHd = NaN;
        bestId = '';
        bestGalleryPath = '';
        compTimes = zeros(1, numel(gallery));
        for g = 1:numel(gallery)
            grec = gallery(g).record;
            t = tic;
            hd = gethammingdistance(probe.template, probe.mask, grec.template, grec.mask, 1);
            compTimes(g) = toc(t);
            if ~isnan(hd) && (isnan(bestHd) || hd < bestHd)
                bestHd = hd;
                bestId = gallery(g).identity;
                bestGalleryPath = grec.rel_path;
            end
        end
        correct = strcmp(bestId, probe.identity);

        r.probe_index = p;
        r.probe_path = string(probe.image_path);
        r.probe_rel_path = string(probe.rel_path);
        r.true_identity = string(probe.identity);
        r.predicted_identity = string(bestId);
        r.correct = double(correct);
        r.best_hamming_distance = bestHd;
        r.best_gallery_rel_path = string(bestGalleryPath);
        r.num_gallery_comparisons = numel(gallery);
        r.matching_total_s = toc(tTotal);
        r.matching_mean_single_comparison_s = mean(compTimes);
        r.matching_min_single_comparison_s = min(compTimes);
        r.matching_max_single_comparison_s = max(compTimes);

        if isempty(idRows); idRows = r; else; idRows(end+1) = r; end %#ok<AGROW>

        if p == 1 || mod(p,50) == 0 || p == numel(probes)
            fprintf('[IDENTYFIKACJA MATLAB] probe %d/%d\n', p, numel(probes));
        end
    end

    if ~isempty(idRows)
        writetable(struct2table(idRows), fullfile(datasetOut, 'identification_top1_results.csv'));
    else
        fid = fopen(fullfile(datasetOut, 'identification_top1_results.csv'), 'w');
        fprintf(fid, 'no_probe_images\n'); fclose(fid);
    end
end

function summary = buildSummaryMatlab(datasetName, rows, idRows, tp, fp, fn, tn, badIouThreshold, badDiceThreshold)
    statuses = string({rows.status});
    okRows = startsWith(statuses, 'ok');
    failedRows = statuses == 'failed';
    iouVals = [rows.iou];
    diceVals = [rows.dice];
    validGt = ~isnan(iouVals);
    badSeg = validGt & (iouVals < badIouThreshold | diceVals < badDiceThreshold);

    summary = struct();
    summary.dataset = datasetName;
    summary.number_of_images_found = numel(rows);
    summary.number_of_images_processed_ok = nnz(okRows);
    summary.number_of_processing_failures = nnz(failedRows);
    summary.number_of_images_with_gt_metrics = nnz(validGt);
    summary.number_of_bad_segmentations_saved = nnz(badSeg);
    summary.bad_iou_threshold = badIouThreshold;
    summary.bad_dice_threshold = badDiceThreshold;

    summary.pupil_iou_mean = meanOmitNan(iouVals);
    summary.pupil_iou_std = stdOmitNan(iouVals);
    summary.pupil_dice_mean = meanOmitNan(diceVals);
    summary.pupil_dice_std = stdOmitNan(diceVals);
    summary.pupil_pixel_accuracy_mean = meanOmitNan([rows.pixel_accuracy]);
    summary.pupil_precision_mean = meanOmitNan([rows.precision]);
    summary.pupil_recall_mean = meanOmitNan([rows.recall]);
    summary.pupil_specificity_mean = meanOmitNan([rows.specificity]);

    epsVal = 1e-12;
    summary.global_tp = tp; summary.global_fp = fp; summary.global_fn = fn; summary.global_tn = tn;
    summary.global_pupil_iou = tp / (tp + fp + fn + epsVal);
    summary.global_pupil_dice = 2 * tp / (2 * tp + fp + fn + epsVal);
    summary.global_pupil_pixel_accuracy = (tp + tn) / (tp + fp + fn + tn + epsVal);

    timeNames = {'read_s','inner_pupil_s','outer_iris_s','top_eyelid_s','bottom_eyelid_s','eyelashes_s','segmentation_total_s','pupil_mask_s','gt_load_s','metrics_s','normalization_s','encoding_s','feature_total_s'};
    for k = 1:numel(timeNames)
        nm = timeNames{k};
        vals = [rows.(nm)];
        summary.([nm '_mean']) = meanOmitNan(vals);
        summary.([nm '_std']) = stdOmitNan(vals);
        summary.([nm '_sum']) = sumOmitNan(vals);
    end

    if ~isempty(idRows)
        correct = sum([idRows.correct]);
        total = numel(idRows);
        summary.identification_num_probes = total;
        summary.identification_correct_top1 = correct;
        summary.identification_top1_accuracy = correct / max(total, 1);
        summary.matching_total_s_mean = meanOmitNan([idRows.matching_total_s]);
        summary.matching_total_s_sum = sumOmitNan([idRows.matching_total_s]);
        summary.matching_mean_single_comparison_s_mean = meanOmitNan([idRows.matching_mean_single_comparison_s]);
    else
        summary.identification_num_probes = 0;
        summary.identification_correct_top1 = 0;
        summary.identification_top1_accuracy = NaN;
        summary.matching_total_s_mean = NaN;
        summary.matching_total_s_sum = NaN;
        summary.matching_mean_single_comparison_s_mean = NaN;
    end
end

function writeSummaryTxtMatlab(path, s)
    fid = fopen(path, 'w');
    fprintf(fid, 'PODSUMOWANIE EKSPERYMENTU CASIA V1 — MASEK/KOVESI MATLAB + PUPIL GT\n');
    fprintf(fid, '==========================================================================================\n');
    fprintf(fid, 'Dataset: %s\n', s.dataset);
    fprintf(fid, 'Liczba obrazów: %d\n', s.number_of_images_found);
    fprintf(fid, 'Przetworzone OK: %d\n', s.number_of_images_processed_ok);
    fprintf(fid, 'Błędy przetwarzania: %d\n', s.number_of_processing_failures);
    fprintf(fid, 'Obrazy z metrykami GT: %d\n', s.number_of_images_with_gt_metrics);
    fprintf(fid, 'Zapisane błędne segmentacje: %d\n', s.number_of_bad_segmentations_saved);
    fprintf(fid, '\n[SEGMENTACJA ŹRENICY VS GT]\n');
    fprintf(fid, 'IoU mean=%.8f std=%.8f\n', s.pupil_iou_mean, s.pupil_iou_std);
    fprintf(fid, 'Dice mean=%.8f std=%.8f\n', s.pupil_dice_mean, s.pupil_dice_std);
    fprintf(fid, 'Pixel accuracy mean=%.8f\n', s.pupil_pixel_accuracy_mean);
    fprintf(fid, 'Precision mean=%.8f\n', s.pupil_precision_mean);
    fprintf(fid, 'Recall mean=%.8f\n', s.pupil_recall_mean);
    fprintf(fid, 'Specificity mean=%.8f\n', s.pupil_specificity_mean);
    fprintf(fid, 'global_pupil_iou=%.8f\n', s.global_pupil_iou);
    fprintf(fid, 'global_pupil_dice=%.8f\n', s.global_pupil_dice);
    fprintf(fid, 'global_pupil_pixel_accuracy=%.8f\n', s.global_pupil_pixel_accuracy);
    fprintf(fid, '\n[CZASY ETAPÓW — SEKUNDY]\n');
    names = fieldnames(s);
    for i = 1:numel(names)
        nm = names{i};
        if endsWith(nm, '_s_mean') || endsWith(nm, '_s_std') || endsWith(nm, '_s_sum')
            fprintf(fid, '%s = %.8f\n', nm, s.(nm));
        end
    end
    fprintf(fid, '\n[IDENTYFIKACJA TOP-1]\n');
    fprintf(fid, 'Probe: %d\n', s.identification_num_probes);
    fprintf(fid, 'Correct: %.0f\n', s.identification_correct_top1);
    fprintf(fid, 'Top-1 accuracy: %.8f\n', s.identification_top1_accuracy);
    fprintf(fid, 'matching_total_s_mean=%.8f\n', s.matching_total_s_mean);
    fprintf(fid, 'matching_total_s_sum=%.8f\n', s.matching_total_s_sum);
    fclose(fid);
end

function row = emptyImageRow()
    % Jednolity zestaw pól, aby struct2table działał nawet dla błędów.
    nanVal = NaN;
    row.dataset = string(''); row.index = nanVal; row.image_path = string(''); row.rel_path = string('');
    row.filename = string(''); row.stem = string(''); row.identity = string(''); row.status = string('');
    row.error = string(''); row.gt_path = string(''); row.gt_note = string(''); row.bad_segmentation = string('');
    row.read_s = nanVal; row.inner_pupil_s = nanVal; row.outer_iris_s = nanVal; row.top_eyelid_s = nanVal; row.bottom_eyelid_s = nanVal;
    row.eyelashes_s = nanVal; row.segmentation_total_s = nanVal; row.pupil_mask_s = nanVal; row.gt_load_s = nanVal; row.metrics_s = nanVal;
    row.normalization_s = nanVal; row.encoding_s = nanVal; row.feature_total_s = nanVal;
    row.pupil_y = nanVal; row.pupil_x = nanVal; row.pupil_r = nanVal; row.iris_y = nanVal; row.iris_x = nanVal; row.iris_r = nanVal;
    row.pred_pupil_area_px = nanVal; row.pred_iris_annulus_area_px = nanVal; row.pred_pupil_mask_path = string(''); row.gt_area_px = nanVal;
    row.tp = nanVal; row.fp = nanVal; row.fn = nanVal; row.tn = nanVal;
    row.iou = nanVal; row.dice = nanVal; row.pixel_accuracy = nanVal; row.precision = nanVal; row.recall = nanVal; row.specificity = nanVal; row.fpr = nanVal; row.fnr = nanVal;
end

function files = findImagesRecursive(rootDir, extensions)
    files = {};
    if exist(rootDir, 'dir') ~= 7
        return;
    end
    listing = dir(rootDir);
    for i = 1:numel(listing)
        name = listing(i).name;
        if strcmp(name, '.') || strcmp(name, '..')
            continue;
        end
        fullp = fullfile(listing(i).folder, name);
        if listing(i).isdir
            sub = findImagesRecursive(fullp, extensions);
            files = [files, sub]; %#ok<AGROW>
        else
            [~,~,ext] = fileparts(name);
            if any(strcmpi(ext, extensions))
                files{end+1} = fullp; %#ok<AGROW>
            end
        end
    end
    files = sort(files);
end

function gtMap = buildGtMap(gtRoot)
    gtMap = containers.Map('KeyType','char','ValueType','char');
    if exist(gtRoot, 'dir') ~= 7
        return;
    end
    exts = {'.png','.bmp','.jpg','.jpeg','.tif','.tiff'};
    files = findImagesRecursive(gtRoot, exts);
    for i = 1:numel(files)
        [~, stem, ~] = fileparts(files{i});
        key = cleanStemMatlab(stem);
        if ~isKey(gtMap, key)
            gtMap(key) = files{i};
        end
        key2 = lower(stem);
        if ~isKey(gtMap, key2)
            gtMap(key2) = files{i};
        end
    end
end

function key = cleanStemMatlab(stem)
    key = lower(stem);
    suffixes = {'_pupil_mask','_mask_pupil','_pupil_gt','_gt_pupil','_pseudo_gt','_pseudo_ground_truth','_ground_truth','_pupil','_mask','_gt','_sobel_hough'};
    changed = true;
    while changed
        changed = false;
        for i = 1:numel(suffixes)
            suf = suffixes{i};
            if strlength(key) > strlength(suf) && endsWith(key, suf)
                key = extractBefore(key, strlength(key)-strlength(suf)+1);
                key = char(key);
                changed = true;
            end
        end
    end
    key = char(key);
end

function gtPath = locateGtMaskMatlab(stem, gtMap)
    keys = {cleanStemMatlab(stem), lower(stem)};
    gtPath = string('');
    for i = 1:numel(keys)
        if isKey(gtMap, keys{i})
            gtPath = string(gtMap(keys{i}));
            return;
        end
    end
end

function [gtMask, note] = loadGtMaskMatlab(maskPath, targetShape)
    gt = imread(maskPath);
    if ndims(gt) == 3
        gt = rgb2gray(gt);
    end
    note = 'original_shape';
    if size(gt,1) ~= targetShape(1) || size(gt,2) ~= targetShape(2)
        gt = imresize(gt, [targetShape(1), targetShape(2)], 'nearest');
        note = 'resized_to_image_shape';
    end
    gtMask = gt > 0;
    fgRatio = nnz(gtMask) / numel(gtMask);
    if fgRatio > 0.50
        gtMask = ~gtMask;
        note = [note ';inverted_auto'];
    end
end

function mask = circleMask(shapeVal, circle)
    h = shapeVal(1); w = shapeVal(2);
    [xx, yy] = meshgrid(1:w, 1:h);
    y0 = double(circle(1)); x0 = double(circle(2)); r = double(circle(3));
    mask = ((yy - y0).^2 + (xx - x0).^2) <= r.^2;
end

function met = binaryMetricsMatlab(pred, gt)
    pred = logical(pred); gt = logical(gt);
    tp = nnz(pred & gt);
    fp = nnz(pred & ~gt);
    fn = nnz(~pred & gt);
    tn = nnz(~pred & ~gt);
    epsVal = 1e-12;
    met.tp = tp; met.fp = fp; met.fn = fn; met.tn = tn;
    met.iou = tp / (tp + fp + fn + epsVal);
    met.dice = 2 * tp / (2 * tp + fp + fn + epsVal);
    met.pixel_accuracy = (tp + tn) / (tp + fp + fn + tn + epsVal);
    met.precision = tp / (tp + fp + epsVal);
    met.recall = tp / (tp + fn + epsVal);
    met.specificity = tn / (tn + fp + epsVal);
    met.fpr = fp / (fp + tn + epsVal);
    met.fnr = fn / (fn + tp + epsVal);
end

function out = makeDiagnosticImageMatlab(img, pred, gt, circlepupil, circleiris, textLines)
    if ndims(img) == 3
        img = rgb2gray(img);
    end
    img = uint8(img);
    base = repmat(img, [1 1 3]);
    panel1 = base;
    panel2 = base;
    panel3 = base;
    panel4 = uint8(double(base) * 0.35);

    if ~isempty(circleiris)
        panel1 = drawCircleRGB(panel1, circleiris, [0 255 255]);
    end
    if ~isempty(circlepupil)
        panel1 = drawCircleRGB(panel1, circlepupil, [255 0 0]);
    end

    if ~isempty(pred)
        panel2 = overlayMaskRGB(panel2, pred, [255 0 0], 0.35);
        panel2 = drawContourRGB(panel2, pred, [255 0 0]);
    end
    if ~isempty(gt)
        panel3 = overlayMaskRGB(panel3, gt, [0 255 0], 0.35);
        panel3 = drawContourRGB(panel3, gt, [0 255 0]);
    end
    if ~isempty(pred) && ~isempty(gt)
        tp = pred & gt; fp = pred & ~gt; fn = ~pred & gt;
        panel4 = setColor(panel4, tp, [0 180 0]);
        panel4 = setColor(panel4, fp, [255 0 0]);
        panel4 = setColor(panel4, fn, [0 0 255]);
    end

    out = [panel1 panel2 panel3 panel4];
    infoH = 25 + 18 * min(numel(textLines), 8);
    info = zeros(infoH, size(out,2), 3, 'uint8');
    % Bez zależności od Computer Vision Toolbox nie rysujemy tekstu bezpośrednio.
    % Informacja tekstowa jest zapisana w nazwie pliku i tabeli CSV.
    out = [out; info];
end

function img = overlayMaskRGB(img, mask, color, alpha)
    for c = 1:3
        ch = img(:,:,c);
        ch(mask) = uint8((1-alpha)*double(ch(mask)) + alpha*color(c));
        img(:,:,c) = ch;
    end
end

function img = setColor(img, mask, color)
    for c = 1:3
        ch = img(:,:,c);
        ch(mask) = uint8(color(c));
        img(:,:,c) = ch;
    end
end

function img = drawContourRGB(img, mask, color)
    contour = contourFromMask(mask);
    img = setColor(img, contour, color);
end

function contour = contourFromMask(mask)
    mask = logical(mask);
    up = circshift(mask, [1,0]); down = circshift(mask, [-1,0]); left = circshift(mask, [0,1]); right = circshift(mask, [0,-1]);
    interior = mask & up & down & left & right;
    contour = mask & ~interior;
    contour(1,:) = mask(1,:); contour(end,:) = mask(end,:); contour(:,1) = mask(:,1); contour(:,end) = mask(:,end);
end

function img = drawCircleRGB(img, circle, color)
    h = size(img,1); w = size(img,2);
    y0 = double(circle(1)); x0 = double(circle(2)); r = double(circle(3));
    theta = linspace(0, 2*pi, 720);
    xs = round(x0 + r * cos(theta));
    ys = round(y0 + r * sin(theta));
    valid = xs >= 1 & xs <= w & ys >= 1 & ys <= h;
    idx = sub2ind([h,w], ys(valid), xs(valid));
    for c = 1:3
        tmp = img(:,:,c);
        tmp(idx) = uint8(color(c));
        img(:,:,c) = tmp;
    end
end

function id = identityFromFilename(stem, imgPath)
    tok = regexp(stem, '^(\d{1,4})[_\-]', 'tokens', 'once');
    if ~isempty(tok)
        id = sprintf('%03d', str2double(tok{1}));
        return;
    end
    [parent,~,~] = fileparts(imgPath);
    [~, parentName, ~] = fileparts(parent);
    tok2 = regexp(parentName, '(\d{1,4})', 'tokens', 'once');
    if ~isempty(tok2)
        id = sprintf('%03d', str2double(tok2{1}));
    else
        id = parentName;
    end
end

function rel = relativePath(pathValue, rootDir)
    rel = strrep(pathValue, [rootDir filesep], '');
    if strcmp(rel, pathValue)
        rel = pathValue;
    end
end

function ensureDir(pathValue)
    if exist(pathValue, 'dir') ~= 7
        mkdir(pathValue);
    end
end

function v = meanOmitNan(x)
    x = x(~isnan(x));
    if isempty(x); v = NaN; else; v = mean(x); end
end

function v = stdOmitNan(x)
    x = x(~isnan(x));
    if isempty(x); v = NaN; else; v = std(x); end
end

function v = sumOmitNan(x)
    x = x(~isnan(x));
    if isempty(x); v = NaN; else; v = sum(x); end
end

function makePlotsMatlab(datasetOut, rows)
    try
        keys = {'iou','dice','segmentation_total_s','feature_total_s'};
        titles = {'IoU pupil','Dice pupil','Segmentation total [s]','Feature extraction total [s]'};
        for k = 1:numel(keys)
            vals = [rows.(keys{k})];
            vals = vals(~isnan(vals));
            if isempty(vals); continue; end
            fig = figure('Visible','off');
            histogram(vals, 30);
            title(titles{k}); xlabel(keys{k}); ylabel('liczba obrazów');
            saveas(fig, fullfile(datasetOut, ['hist_' keys{k} '.png']));
            close(fig);
        end
    catch
    end
end
