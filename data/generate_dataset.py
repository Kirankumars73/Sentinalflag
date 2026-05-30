"""
SentinelFlag — Dataset Generator
Generates labelled training data for 5 sensitivity classes.

Classes:
  0 = Non-sensitive
  1 = PII (name, email, phone, national ID)
  2 = Financial (card numbers, IBAN, amounts, transactions)
  3 = Health / Medical
  4 = Location-critical (precise GPS, home address, real-time whereabouts)

Run:
  pip install faker pandas scikit-learn
  python generate_dataset.py

Output:
  train.csv, val.csv, test.csv  (in ./output/)
"""

import random
import csv
import os
from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLES_PER_CLASS = 2000   # increase to 10000+ for production training


# ── Class 0: Non-sensitive ─────────────────────────────────────────────────
def gen_non_sensitive():
    templates = [
        lambda: fake.sentence(nb_words=random.randint(8, 20)),
        lambda: f"The weather today is {random.choice(['sunny','cloudy','rainy','windy'])} with a high of {random.randint(10,35)}°C.",
        lambda: f"I really enjoyed the movie last night. {fake.sentence()}",
        lambda: f"Meeting scheduled for {random.choice(['Monday','Tuesday','Wednesday','Thursday','Friday'])} at {random.randint(9,17)}:00.",
        lambda: f"Please review the attached document and provide feedback by end of week.",
        lambda: f"The project deadline has been moved to {random.choice(['next week','end of month','Q3'])}.",
        lambda: f"Can you help me with the {random.choice(['report','presentation','analysis','summary'])}?",
        lambda: f"I went to the {random.choice(['supermarket','gym','library','park','office'])} today.",
        lambda: f"Reminder: team lunch on {fake.day_of_week()} at noon.",
        lambda: f"{fake.catch_phrase()}. {fake.sentence()}",
        lambda: f"The new software update includes {random.randint(3,15)} bug fixes and performance improvements.",
        lambda: f"Looking forward to the weekend! Planning to {random.choice(['cook','read','hike','relax','visit family'])}.",
        lambda: f"The quarterly review went well. {fake.sentence()}",
        lambda: f"Please find the minutes from today's standup below.",
        lambda: f"Happy birthday to {fake.first_name()}! 🎂",
    ]
    return random.choice(templates)()


# ── Class 1: PII ──────────────────────────────────────────────────────────
def gen_pii():
    templates = [
        lambda: f"My name is {fake.name()} and my email is {fake.email()}.",
        lambda: f"Please contact {fake.name()} at {fake.phone_number()}.",
        lambda: f"User profile: {fake.name()}, DOB: {fake.date_of_birth(minimum_age=18, maximum_age=80).strftime('%d/%m/%Y')}, email: {fake.email()}",
        lambda: f"Passport number: {fake.bothify(text='??#######').upper()} — {fake.name()}",
        lambda: f"National ID: {fake.ssn()} for {fake.name()}",
        lambda: f"Driver's license: {fake.bothify(text='??-####-####')} issued to {fake.name()}",
        lambda: f"Send the invoice to {fake.name()} at {fake.email()} or call {fake.phone_number()}.",
        lambda: f"My username is {fake.user_name()} and I registered with {fake.email()}.",
        lambda: f"Contact info — Name: {fake.name()}, Phone: {fake.phone_number()}, Email: {fake.email()}",
        lambda: f"The account belongs to {fake.name()}, SSN: {fake.ssn()}.",
        lambda: f"Emergency contact: {fake.name()}, relationship: {random.choice(['spouse','parent','sibling','friend'])}, phone: {fake.phone_number()}",
        lambda: f"Date of birth: {fake.date_of_birth(minimum_age=18, maximum_age=80).strftime('%B %d, %Y')}. Full name: {fake.name()}.",
        lambda: f"New user registration — {fake.name()}, {fake.email()}, {fake.phone_number()}",
        lambda: f"Please update my email from {fake.email()} to {fake.email()}. — {fake.name()}",
        lambda: f"Voter ID: {fake.bothify(text='VID-########')} — {fake.name()}, DOB: {fake.date_of_birth().strftime('%Y-%m-%d')}",
    ]
    return random.choice(templates)()


# ── Class 2: Financial ────────────────────────────────────────────────────
def gen_financial():
    def card():
        return " ".join([fake.bothify(text="####") for _ in range(4)])
    def iban():
        return fake.bothify(text="GB##BARC########").upper()
    def amount():
        return f"${random.randint(10, 50000):,.2f}"

    templates = [
        lambda: f"Card number: {card()}, CVV: {random.randint(100,999)}, expiry: {fake.credit_card_expire()}",
        lambda: f"Please transfer {amount()} to IBAN {iban()}.",
        lambda: f"Transaction ID: {fake.uuid4()[:8].upper()} — amount: {amount()} deducted from account.",
        lambda: f"Your bank account ending in {random.randint(1000,9999)} has a balance of {amount()}.",
        lambda: f"Invoice #{random.randint(10000,99999)}: Total due {amount()}. Payment via card {card()}.",
        lambda: f"Wire transfer of {amount()} to account {fake.bban()} processed successfully.",
        lambda: f"Crypto wallet: {fake.sha256()[:34]} — balance: {random.uniform(0.001, 10.0):.4f} BTC",
        lambda: f"Routing number: {fake.bothify(text='#########')}, Account: {fake.bban()}",
        lambda: f"Salary credit of {amount()} received from {fake.company()}.",
        lambda: f"Your credit card limit is {amount()}. Current outstanding: {amount()}.",
        lambda: f"PayPal payment of {amount()} sent to {fake.email()}.",
        lambda: f"Stock portfolio value: {amount()}. Holdings: {random.randint(5,50)} shares of {fake.company()}.",
        lambda: f"Loan account #{fake.bothify(text='LN-#######')} — outstanding balance {amount()} at {random.uniform(3,15):.1f}% APR.",
        lambda: f"Refund of {amount()} issued to card ending {random.randint(1000,9999)}.",
        lambda: f"Tax reference: {fake.bothify(text='UTR-##########')} — self-assessment due: {amount()}",
    ]
    return random.choice(templates)()


# ── Class 3: Health / Medical ─────────────────────────────────────────────
def gen_health():
    conditions = ["diabetes", "hypertension", "asthma", "depression", "anxiety disorder",
                  "HIV", "cancer", "ADHD", "bipolar disorder", "chronic kidney disease",
                  "heart disease", "epilepsy", "multiple sclerosis", "lupus", "PTSD"]
    medications = ["metformin", "lisinopril", "sertraline", "atorvastatin", "omeprazole",
                   "amoxicillin", "ibuprofen 800mg", "insulin", "levothyroxine", "alprazolam"]
    procedures = ["blood test", "MRI scan", "chemotherapy", "colonoscopy", "ECG",
                  "biopsy", "dialysis", "appendectomy", "hip replacement", "CT scan"]

    templates = [
        lambda: f"Patient diagnosed with {random.choice(conditions)}. Prescribed {random.choice(medications)}.",
        lambda: f"Lab results: HbA1c {random.uniform(4,12):.1f}%, fasting glucose {random.randint(70,300)} mg/dL.",
        lambda: f"I have been taking {random.choice(medications)} for my {random.choice(conditions)} for {random.randint(1,10)} years.",
        lambda: f"Medical history: {random.choice(conditions)}, {random.choice(conditions)}. Current medications: {random.choice(medications)}.",
        lambda: f"Scheduled {random.choice(procedures)} for {fake.date_this_year().strftime('%d %B')}. Diagnosis: {random.choice(conditions)}.",
        lambda: f"Discharge summary: patient treated for {random.choice(conditions)}. Follow-up in {random.randint(2,8)} weeks.",
        lambda: f"My doctor prescribed {random.choice(medications)} {random.randint(10,500)}mg twice daily.",
        lambda: f"Therapy session notes: patient reports symptoms consistent with {random.choice(conditions)}.",
        lambda: f"Insurance claim for {random.choice(procedures)} — diagnosis code: {fake.bothify(text='?##.#')}",
        lambda: f"Blood pressure: {random.randint(100,160)}/{random.randint(60,100)} mmHg. Heart rate: {random.randint(55,100)} bpm.",
        lambda: f"Allergy alert: patient is allergic to {random.choice(['penicillin','sulfa drugs','latex','aspirin','codeine'])}.",
        lambda: f"Mental health assessment — PHQ-9 score: {random.randint(0,27)}. Diagnosis: {random.choice(conditions)}.",
        lambda: f"Prescription: {random.choice(medications)} — qty: {random.randint(30,90)} tablets, refills: {random.randint(0,5)}.",
        lambda: f"Genetic test result: BRCA{random.randint(1,2)} {random.choice(['positive','negative','variant of uncertain significance'])}.",
        lambda: f"HIV status: {random.choice(['positive','negative'])}. Last tested: {fake.date_this_year().strftime('%B %Y')}.",
    ]
    return random.choice(templates)()


# ── Class 4: Location-critical ────────────────────────────────────────────
def gen_location():
    templates = [
        lambda: f"Current location: {fake.latitude():.6f}, {fake.longitude():.6f}",
        lambda: f"Home address: {fake.street_address()}, {fake.city()}, {fake.postcode()}",
        lambda: f"GPS coordinates: lat {fake.latitude():.5f}, lng {fake.longitude():.5f} — updated {random.randint(1,60)} seconds ago.",
        lambda: f"Delivery address: {fake.address()}. Phone: {fake.phone_number()}.",
        lambda: f"I live at {fake.street_address()}, {fake.city()}. Come over anytime.",
        lambda: f"Real-time location shared: {fake.city()}, near {fake.street_name()} and {fake.street_name()}.",
        lambda: f"Check-in: {fake.local_latlng(country_code='US')[2]}, {fake.street_address()}. Time: {fake.time()}",
        lambda: f"Tracking device last seen at {fake.latitude():.4f}°N {fake.longitude():.4f}°E.",
        lambda: f"She is currently at {fake.street_address()}, {fake.city()} based on her phone location.",
        lambda: f"Route: from {fake.street_address()}, {fake.city()} to {fake.street_address()}, {fake.city()}.",
        lambda: f"My kid's school is at {fake.street_address()}, {fake.city()}. Pick-up at 3pm.",
        lambda: f"Location ping: user at {fake.latitude():.6f}, {fake.longitude():.6f} — {fake.city()} downtown.",
        lambda: f"Geofence alert: target entered zone at {fake.street_address()}, {fake.city()} at {fake.time()}.",
        lambda: f"I'm staying at {fake.street_address()}, {fake.city()} this week. Room {random.randint(100,999)}.",
        lambda: f"Precise location: {fake.coordinate()}, altitude {random.randint(0,3000)}m — live tracking active.",
    ]
    return random.choice(templates)()


# ── Build and write datasets ───────────────────────────────────────────────
generators = [
    gen_non_sensitive,
    gen_pii,
    gen_financial,
    gen_health,
    gen_location,
]

print(f"Generating {SAMPLES_PER_CLASS} samples per class ({SAMPLES_PER_CLASS * 5} total)...")

all_samples = []
for label, gen in enumerate(generators):
    for _ in range(SAMPLES_PER_CLASS):
        text = gen()
        # Clean up: strip newlines, limit length
        text = text.replace("\n", " ").strip()[:512]
        all_samples.append({"text": text, "label": label})

random.shuffle(all_samples)

# 70/15/15 split
n = len(all_samples)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

splits = {
    "train": all_samples[:train_end],
    "val":   all_samples[train_end:val_end],
    "test":  all_samples[val_end:],
}

for split_name, rows in splits.items():
    path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ {split_name}.csv — {len(rows)} samples → {path}")

# Class distribution check
from collections import Counter
train_labels = [r["label"] for r in splits["train"]]
counts = Counter(train_labels)
label_names = ["non-sensitive", "PII", "financial", "health", "location"]
print("\nTrain set class distribution:")
for i, name in enumerate(label_names):
    print(f"  Class {i} ({name}): {counts[i]} samples")

print("\nDataset generation complete.")
