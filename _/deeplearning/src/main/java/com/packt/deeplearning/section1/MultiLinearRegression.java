package com.packt.deeplearning.section1;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.converters.SelfWritableConverter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.IOException;
import java.text.DecimalFormat;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;

public class MultiLinearRegression {
    private final static Logger LOGGER = org.slf4j.LoggerFactory.getLogger(LogisticRegression.class);

    public static void main(String[] args) throws IOException, InterruptedException {
        BasicConfigurator.configure();
        DecimalFormat f = new DecimalFormat("#0.000");
        int numLinesToSkip = 1;
        String delimiter = ",";
        RecordReader trainReader = new CSVRecordReader(numLinesToSkip, delimiter);
        trainReader.initialize(new FileSplit(new ClassPathResource("data/multilinearregression/train.csv").getFile()));
        RecordReader testReader = new CSVRecordReader(numLinesToSkip, delimiter);
        testReader.initialize(new FileSplit(new ClassPathResource("data/multilinearregression/test.csv").getFile()));
        int labelIndex = 6;
        int numClasses = -1;
        int batchSize = 1024;
        double learningRate = 1e-2;
        int epoch = 5;

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainReader, new SelfWritableConverter(), batchSize, labelIndex, numClasses, true);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testReader, new SelfWritableConverter(), batchSize, labelIndex, numClasses, true);

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainIter);
        trainIter.reset();
        trainIter.setPreProcessor(normalizer);
        testIter.setPreProcessor(normalizer);

        final int numInputs = trainIter.inputColumns();
        int outputNum = 1;
        int iterations = 1;
        long seed = 6;

        LOGGER.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.95)
                .dropOut(0.5)
                .learningRate(learningRate)
                .list()
                .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.RELU).nIn(trainIter.inputColumns()).nOut(1).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1000));


        //Train the network on the full data set, and evaluate in periodically
        Instant start = Instant.now();
        for (int i = 0; i < epoch; i++) {
            trainIter.reset();
            testIter.reset();
            net.fit(trainIter);
            LOGGER.info("Epoch " + i + " complete. Evaluation:");
            RegressionEvaluation eval = new RegressionEvaluation(Arrays.asList("signals"), 4);
            while (testIter.hasNext()) {
                DataSet t = testIter.next();
                INDArray features = t.getFeatures();
                INDArray labels = t.getLabels();
                INDArray predicted = net.output(features, false);
                eval.eval(labels, predicted);
            }
            LOGGER.info("RMSE: " + f.format(eval.rootMeanSquaredError(0)));
        }
        Instant end = Instant.now();
        LOGGER.info("Evaluated in {}  ms", Duration.between(end, start).toMillis());
    }
}
