import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Fait en sorte que seulement les attributs utilises pour l'entrainement 
 * soit utilise pour l'evaluation.
 */
public class AttributeClassifier extends Classifier {
	private static final long serialVersionUID = -513615971340002718L;
	
	private Classifier classifier;
	private List<String> attributes;
	private Instances dataset;
	
	public AttributeClassifier(Classifier classifier) {
		this.classifier = classifier;
	}
	
	public void buildClassifier(Instances inst) throws Exception {
		int nbAttr = inst.numAttributes();
		String[] attributes = new String[nbAttr];
		
		for (int i=0; i<nbAttr; i++) {
			attributes[i] = inst.attribute(i).name();
		}
		
		this.attributes = Arrays.asList(attributes);
		this.classifier.buildClassifier(inst);
	}

	public double classifyInstance(Instance instP) throws Exception {
		Instance inst = new Instance(instP);
		boolean initDataset = this.dataset == null;
		
		if (initDataset) {
			this.dataset = new Instances(instP.dataset());
		}
		
		for (int i=inst.numAttributes() - 1; i>=0; i--) {
			if (!this.attributes.contains(instP.attribute(i).name())) {
				inst.deleteAttributeAt(i);
				
				if (initDataset) {
					this.dataset.deleteAttributeAt(i);
				}
			}
		}
		
		inst.setDataset(this.dataset);
		return this.classifier.classifyInstance(inst);
	}
	
	
	public String debugTipText() {
		return this.classifier.debugTipText();
	}

	public double[] distributionForInstance(Instance instP) throws Exception {
		Instance inst = new Instance(instP);
		boolean initDataset = this.dataset == null;
		
		if (initDataset) {
			this.dataset = new Instances(instP.dataset());
		}
		
		for (int i=inst.numAttributes() - 1; i>=0; i--) {
			if (!this.attributes.contains(instP.attribute(i).name())) {
				inst.deleteAttributeAt(i);
				
				if (initDataset) {
					this.dataset.deleteAttributeAt(i);
				}
			}
		}
		
		inst.setDataset(this.dataset);
		return this.classifier.distributionForInstance(inst);
	}

	public Capabilities getCapabilities() {
		return this.classifier.getCapabilities();
	}

	public boolean getDebug() {
		return this.classifier.getDebug();
	}
	
	public String[] getOptions() {
		return this.classifier.getOptions();
	}
	
	public String getRevision() {
		return this.classifier.getRevision();
	}

	@SuppressWarnings("rawtypes")
	public Enumeration listOptions() {
		return this.classifier.listOptions();
	}

	public void setDebug(boolean debug) {
		this.classifier.setDebug(debug);
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		return super.clone();
	}
	

}
