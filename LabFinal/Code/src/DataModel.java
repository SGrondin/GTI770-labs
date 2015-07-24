import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


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
	}
}
