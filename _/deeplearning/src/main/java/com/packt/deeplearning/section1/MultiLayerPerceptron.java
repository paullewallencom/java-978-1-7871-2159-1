package com.packt.deeplearning.section1;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;

public class MultiLayerPerceptron {
    private final static Logger LOGGER = org.slf4j.LoggerFactory.getLogger(MultiLayerPerceptron.class);

    public static void main(String[] args) throws IOException, InterruptedException {
        BasicConfigurator.configure();
        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader allDataCsv = new CSVRecordReader(numLinesToSkip, delimiter);
        allDataCsv.initialize(new FileSplit(new ClassPathResource("data/multilayerperceptron/IRIS_.csv").getFile()));

        int labelIndex = 4;
        int numClasses = 3;
        int batchSize = 150;
        int seed = 30;
        DataSetIterator allDataIterator = new RecordReaderDataSetIterator(allDataCsv, batchSize, labelIndex, numClasses);

        org.nd4j.linalg.dataset.DataSet allData = allDataIterator.next();
        allData.normalize();
        allData.shuffle(seed);
        SplitTestAndTrain splitTestAndTrain = allData.splitTestAndTrain(0.80);

        org.nd4j.linalg.dataset.DataSet train = splitTestAndTrain.getTrain();
        org.nd4j.linalg.dataset.DataSet test = splitTestAndTrain.getTest();

        double learningRate = 0.001;
        int numInput = allDataIterator.inputColumns();
        int numOutputs = 3;
        int nHidden = 20;
        int epoch = 10000;
        int iterations = 1;
        LOGGER.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS)
                .momentum(0.9)
                .learningRate(learningRate)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden).activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nIn(nHidden).nOut(nHidden).activation(Activation.RELU).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nOut(numOutputs).nIn(nHidden).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1000));


        Instant start = Instant.now();

        for (int i = 0; i < epoch; i++) {
            net.fit(train);
            LOGGER.info("Epoch " + i + " complete. Evaluation:");
            Evaluation eval = new Evaluation(Arrays.asList("Iris-versicolor", "Iris-virginica", "Iris-setosa"), 3);
            INDArray features = test.getFeatures();
            INDArray labels = test.getLabels();
            INDArray predicted = net.output(features, false);
            eval.eval(labels, predicted);
            LOGGER.info(eval.stats());
        }
        Instant end = Instant.now();
        LOGGER.info("Evaluated in {}  ms", Duration.between(end, start).toMillis());
    }
}
