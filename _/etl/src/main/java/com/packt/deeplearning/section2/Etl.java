package com.packt.deeplearning.section2;

import com.google.common.io.Files;
import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.xml.DOMConfigurator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.AnalyzeSpark;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.slf4j.Logger;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.List;

public class Etl {
    private final static Logger LOGGER = org.slf4j.LoggerFactory.getLogger(Etl.class);
    public static void main(String[] args) throws IOException, InterruptedException {
        DOMConfigurator.configure("src/main/resources/log4j.xml");
        File inputFile = new ClassPathResource("data/multilayerperceptron/iris.csv").getFile();
        Schema inputSchema = new Schema.Builder()
                .addColumnInteger("Id")
                .addColumnDouble("SepalLengthCm")
                .addColumnDouble("SepalWidthCm")
                .addColumnDouble("PetalLengthCm")
                .addColumnDouble("PetalWidthCm")
                .addColumnCategorical("Species", Arrays.asList("Iris-versicolor", "Iris-virginica", "Iris-setosa"))
                .build();
        TransformProcess tP = new TransformProcess.Builder(inputSchema).removeColumns("Id").categoricalToInteger("Species").build();
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("Iris Etl");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        JavaRDD<String> lines = sc.textFile(inputFile.getPath());

        RecordReader allDataCsv = new CSVRecordReader();
        JavaRDD<List<Writable>> parsedData = lines.map(new StringToWritablesFunction(allDataCsv));
        int maxHistogramBuckets = 10;
        DataAnalysis dataAnalysis = AnalyzeSpark.analyze(inputSchema, parsedData, maxHistogramBuckets);
        System.out.println(dataAnalysis);

/*
        JavaRDD<List<Writable>> processed = SparkTransformExecutor.execute(parsedData, tP);

        BufferedWriter writer = Files.newWriter(new File("IRIS.csv"), Charset.defaultCharset());
        processed.map(new WritablesToStringFunction(","))
                 .collect()
                 .forEach(line -> {
                     try {
                         writer.write(line);
                         writer.newLine();
                     }catch (IOException e){
                         LOGGER.error("{}",e);
                     }
                 });
        writer.close();*/
    }
}
