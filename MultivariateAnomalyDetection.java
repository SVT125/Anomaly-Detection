import org.apache.commons.math3.linear.*;
import java.util.*;
import java.io.*;
import org.apache.commons.math3.stat.descriptive.AbstractStorelessUnivariateStatistic;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.stat.correlation.Covariance;

class MultivariateAnomalyDetection extends AnomalyDetection {
	private RealMatrix covariance;
	
	public MultivariateAnomalyDetection(String trainingFileName, String testFileName, double epsilon) {
		super(trainingFileName,testFileName,epsilon);
	}
	
	// Run the multivariate anomaly detection algorithm.
	public boolean[] runMultivariateAlgorithm() throws IOException {
		trainingExamples = readExamples(trainingFileName);
		testExamples = readExamples(testFileName);
		numTestExamples = testExamples.getRowDimension();
		mean = calculateStatistic(trainingExamples, new Mean());
		covariance = calculateCovariance(trainingExamples);
		System.out.println("The given threshold is: " + epsilon);
		return detect(testExamples,mean,covariance,epsilon);		
	}
	
	// Calculates the covariance matrix given the examples.
	private RealMatrix calculateCovariance(RealMatrix examples) {
		return new Covariance(examples).getCovarianceMatrix();
	}
	
	// Runs the anomaly detection algorithm on the test examples.
	private boolean[] detect(RealMatrix examples, RealVector mean, RealMatrix cov, double threshold) {
		boolean[] results = new boolean[numTestExamples];
		final int features = examples.getColumnDimension();
		for( int i = 0; i < numTestExamples; i++ ) {
			RealVector example = examples.getRowVector(i);
				double probability = normalProbability(example.toArray(),mean,cov);
			if(probability < threshold)
				results[i] = true;
			else
				results[i] = false;
		}
		return results;
	}
	
	// Calculates the probability of the given x in the normal distribution N(mean,stddev).
	private double normalProbability(double[] x, RealVector mean, RealMatrix cov) {
		return new MultivariateNormalDistribution(mean.toArray(),cov.getData()).density(x);
	}
}