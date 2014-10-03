// At the moment: instantiate a new class per algorithm test. Here, testing univariate anomaly detection.
// Future notes: All ML algorithm implementations should have a runAlgorithm() method, inherit from a base class?
// Lambda expressions with a functional interface is an option - Function<T,R> object.
import java.io.*;
import java.util.*;	

class OptimizeAlgorithm {
	public boolean[] algorithmResults, realResults;
	public OptimizeAlgorithm(MLAlgorithm mlalg) {
		try {
			this.algorithmResults = mlalg.runAlgorithm();
		}catch(IOException ioe) {
			System.err.println("IOException occured...");
		}
	}
	
	public static void main(String[] args) throws Exception {
		OptimizeAlgorithm oa = new OptimizeAlgorithm(new AnomalyDetection("anomalytraining.txt","anomalytest.txt",.0005));
		oa.realResults = oa.readExamples("trueresults.txt"); // read in the true examples
		double f1Score = oa.evaluateF1Score(oa.algorithmResults,oa. realResults);
		System.out.println(f1Score);
	}
	
	// Read in the true labels of the examples.
	private boolean[] readExamples(String fileName) throws IOException {
		List<Boolean> list = new ArrayList<Boolean>();
		File f = new File(fileName);
		BufferedReader br = new BufferedReader(new FileReader(f));
		String line;
		while((line = br.readLine()) != null) {
			boolean result = Boolean.parseBoolean(line);
			list.add(result);
		}	
		int numResults = list.size();
		boolean[] results = new boolean[numResults];
		for( int i = 0; i < numResults; i++ ) {
			results[i] = list.get(i);
		}
		return results;
	}
	
	// Evaluates the F1 score.
	private double evaluateF1Score(boolean[] algResults, boolean[] realResults) {
		double truePositives = this.intersectionCount(algResults,realResults);
		double precision = truePositives/algResults.length, recall = truePositives/realResults.length;
		System.out.println("The number of true positives was: " + truePositives);
		System.out.println("The precision was: " + precision);
		System.out.println("The recall was: " + recall);
		double f1Score = (2 * precision * recall) / (precision + recall);
		return f1Score;
	}
	
	// Finds the intersection count of the two boolean arrays.
	private int intersectionCount(boolean[] array, boolean[] array2) {
		int intersectionCount = 0;
		if(array.length != array2.length)
			throw new IllegalArgumentException("Unequal array lengths");
		int length = array.length;
		for( int i = 0; i < length; i++ ) {
			if(array[i] != array2[i])
				intersectionCount++;
		}
		return intersectionCount;
	}
}