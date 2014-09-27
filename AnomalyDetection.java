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
	protected double epsilon;
	protected BlockRealMatrix trainingExamples, testExamples;
	protected ArrayRealVector mean, stddev;
	protected int numTestExamples;
	protected String trainingFileName, testFileName;
	
	public AnomalyDetection(String trainingFileName, String testFileName, double epsilon) {
		this.trainingFileName = trainingFileName;
		this.testFileName = testFileName;
		this.epsilon = epsilon;
	}
	
	// Run the anomaly detection algorithm.
	public void runAlgorithm() throws IOException {
		trainingExamples = readExamples(trainingFileName);
		testExamples = readExamples(testFileName);
		numTestExamples = testExamples.getRowDimension();
		mean = calculateStatistic(trainingExamples, new Mean());
		stddev = calculateStatistic(trainingExamples, new StandardDeviation());
		System.out.println("The given threshold is: " + epsilon);
		detect(testExamples,mean,stddev,epsilon);		
	}
	
	// Runs the anomaly detection algorithm on the test examples.
	private void detect(RealMatrix examples, RealVector mean, RealVector stddev, double threshold) {
		final int features = examples.getColumnDimension();
		for( int i = 0; i < numTestExamples; i++ ) {
			double probability = 1;
			RealVector example = examples.getRowVector(i);
			for( int j = 0; j < features; j++ )
				probability = probability * normalProbability(example.getEntry(j),mean.getEntry(j),stddev.getEntry(j));
			if(probability < threshold)
				System.out.println("Example " + (i+1) + " is an anomaly: " + true + ". Probability: " + probability);
			else
				System.out.println("Example " + (i+1) + " is an anomaly: " + false + ". Probability: " + probability);
		}
	}
	
	// Calculates the probability of the given x in the normal distribution N(mean,stddev).
	private double normalProbability(double x, double mean, double stddev) {
		return new NormalDistribution(mean,stddev).density(x);
	}
	
	// Reads in all the examples, given the file name, as a BlockRealMatrix splitting by whitespaces.
	public BlockRealMatrix readExamples(String fileName) throws IOException {
		List<List<Double>> list = new ArrayList<List<Double>>();
		File f = new File(fileName);
		BufferedReader br = new BufferedReader(new FileReader(f));
		String line;
		while((line = br.readLine()) != null) {
			List<Double> convertedStrings = new ArrayList<Double>();
			List<String> strings = Arrays.asList(line.split("\\s+"));
			for( String s : strings )
				convertedStrings.add(Double.parseDouble(s));
				
			list.add(convertedStrings);
		}	
		int numExamples = list.size(), numFeatures = list.get(0).size();
		BlockRealMatrix examples = new BlockRealMatrix(numExamples,numFeatures);
		for( int i = 0; i < numExamples; i++ ) {
			Double[] example = new Double[numFeatures];
			list.get(i).toArray(example);
			examples.setRowVector(i,new ArrayRealVector(example));
		}
		return examples;
	}
	
	// Calculates the given statistic per feature.
	protected ArrayRealVector calculateStatistic(RealMatrix examples, AbstractStorelessUnivariateStatistic statistic) throws IllegalArgumentException {
		int rows = examples.getRowDimension(), cols = examples.getColumnDimension();
		double[] means = new double[cols];
		for( int i = 0; i < cols; i++ ) {
			double[] features = examples.getColumnVector(i).toArray();
			means[i] = statistic.evaluate(features,0,features.length);
		}
		return new ArrayRealVector(means);
	}
}