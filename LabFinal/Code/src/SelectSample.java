import java.io.File;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;


public class SelectSample {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		String inFolder = "../Data/";
		String outFolder = "../Sample/";
		
		// Initialize the filter
		RemovePercentage removeData = new RemovePercentage();
		removeData.setOptions(new String[]{ "-P", "90" });
		
		for (final File fileEntry : new File(inFolder).listFiles()) {
			if (fileEntry.getName().endsWith(".arff")) {
	            String filename = fileEntry.getPath();
	            Instances ds = DataSource.read(filename);
	            removeData.setInputFormat(ds);
	            
	            Instances newData = Filter.useFilter(ds, removeData);
	            
	            DataSink.write(outFolder + fileEntry.getName(), newData);
			}
	    }
	}

}
