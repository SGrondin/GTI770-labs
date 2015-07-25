import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;


public class DataModel {
	public Instances jmirmfccs;
	public Instances marsyas;
	public Instances ssd;
	public Instances rh;
	
	public DataModel() {}
	
	public DataModel(String folder) throws Exception {
		this.jmirmfccs = DataSource.read(folder + "msd-jmirmfccs_dev.arff");
		this.marsyas = DataSource.read(folder + "msd-marsyas_dev_new.arff");
		this.ssd = DataSource.read(folder + "msd-ssd_dev.arff");
		this.rh = DataSource.read(folder + "msd-rh_dev_new.arff");
		
		// On enlève les attributes textes qui ne sont pas pertinents //
		this.jmirmfccs = InstanceUtils.removeIdentifier(this.jmirmfccs);
		this.marsyas = InstanceUtils.removeIdentifier(this.marsyas);
		this.ssd = InstanceUtils.removeIdentifier(this.ssd);
		this.rh = InstanceUtils.removeIdentifier(this.rh);
		
		// On rend unique les noms d'attributs //
		this.jmirmfccs = InstanceUtils.prefixAttributeName("jmirmfccs_", this.jmirmfccs);
		this.marsyas = InstanceUtils.prefixAttributeName("marsyas_", this.rh);
		this.ssd = InstanceUtils.prefixAttributeName("ssd_", this.ssd);
		this.rh = InstanceUtils.prefixAttributeName("rh_", this.rh);
	}
	
	public DataModel getTrainingSet(String percent) throws Exception {
		RemovePercentage removeData = new RemovePercentage();
		removeData.setOptions(new String[]{ "-P", percent });
		
		DataModel newModel = new DataModel();
		
		removeData.setInputFormat(this.jmirmfccs);
		newModel.jmirmfccs = Filter.useFilter(this.jmirmfccs, removeData);
		
		removeData.setInputFormat(this.marsyas);
		newModel.marsyas = Filter.useFilter(this.marsyas, removeData);
		
		removeData.setInputFormat(this.ssd);
		newModel.ssd = Filter.useFilter(this.ssd, removeData);
		
		removeData.setInputFormat(this.rh);
		newModel.rh = Filter.useFilter(this.rh, removeData);
		
		return newModel;
	}
	
	public DataModel getTestSet(String percent) throws Exception {
		RemovePercentage removeData = new RemovePercentage();
		removeData.setOptions(new String[]{ "-P", percent, "-V" });
		
		DataModel newModel = new DataModel();
		
		removeData.setInputFormat(this.jmirmfccs);
		newModel.jmirmfccs = Filter.useFilter(this.jmirmfccs, removeData);
		
		removeData.setInputFormat(this.marsyas);
		newModel.marsyas = Filter.useFilter(this.marsyas, removeData);
		
		removeData.setInputFormat(this.ssd);
		newModel.ssd = Filter.useFilter(this.ssd, removeData);
		
		removeData.setInputFormat(this.rh);
		newModel.rh = Filter.useFilter(this.rh, removeData);
		
		return newModel;
	}
}
