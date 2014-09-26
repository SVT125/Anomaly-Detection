// Anomaly detection algorithm using the univariate normal distribution.
// Assumes all examples are fully defined, that all features per example are set.
// Variance is calculated with denominator of (n-1).
// The example matrix is of dimensions examples x features.

import org.apache.commons.math3.linear.*;
import java.util.*;
import java.io.*;
import org.apache.commons.math3.stat.descriptive.AbstractStorelessUnivariateStatistic;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.distribution.NormalDistribution;

class AnomalyDetection {
	static double epsilon;
	static BlockRealMatrix trainingExamples, testExamples;
	static ArrayRealVector mean, stddev;
	static int numTestExamples;
	public static void main(String[] args) throws IOException {
		trainingExamples = readExamples("anomalytraining.txt");
		testExamples = readExamples("anomalytest.txt");
		numTestExamples = testExamples.getRowDimension();
		mean = calculateStatistic(trainingExamples, new Mean());
		stddev = calculateStatistic(trainingExamples, new StandardDeviation());
		detect(testExamples,mean,stddev);		
	}
	
	public static void detect(BlockRealMatrix examples, ArrayRealVector mean, ArrayRealVector stddev, double threshold) {
		int features = examples.getColumnDimension();
		for( int i = 0; i < numTestExamples; i++ ) {
			double probability = 1;
			RealVector example = examples.getRowVector(i);
			for( int j = features; j > 0; j-- )
				probability = probability * normalProbability(example.getEntry(j),mean,stddev);
			if(probability < threshold)
				System.out.println("Example " + (i+1) + " is an anomaly: " + true);
			else
				System.out.println("Example " + (i+1) + " is an anomaly: " + false);
		}
	}
	
	public static double normalProbability(double x, double mean, double stddev) {
		return new NormalDistribution(mean,stddev).density(x);
	}
	
	public static BlockRealMatrix readExamples(String fileName) throws IOException {
		List<List<Double>> list = new ArrayList<List<Double>>();
		File f = new File(fileName);
		BufferedReader br = new BufferedReader(new FileReader(f));
		String line;
		while((line = br.readLine()) != null) {
			List<Double> convertedStrings = new ArrayList<Double>();
			List<String> strings = Arrays.asList(line.split("//s+"));
			for( String s : strings )
				convertedStrings.add(Double.parseDouble(s));
				
			list.add(convertedStrings);
		}	
		int numExamples = list.size(), numFeatures = list.get(0).size();
		BlockRealMatrix examples = new BlockRealMatrix(numExamples,numFeatures);
		for( int i = 0; i < numExamples; i++ ) {
			Double[] example = (Double[])list.get(i).toArray();
			examples.setRowVector(i,new ArrayRealVector(example));
		}
		return examples;
	}
	
	public static ArrayRealVector calculateStatistic(BlockRealMatrix examples, AbstractStorelessUnivariateStatistic statistic) throws IllegalArgumentException {
		int rows = examples.getRowDimension(), cols = examples.getColumnDimension();
		double[] means = new double[cols];
		for( int i = 0; i < cols; i++ ) {
			double[] features = examples.getColumnVector(cols).toArray();
			means[i] = statistic.evaluate(features,0,features.length);
		}
		return new ArrayRealVector(means);
	}
}