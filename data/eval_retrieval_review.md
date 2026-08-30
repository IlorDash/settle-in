# Group C, manual grounding pass (C3)

One section per query, holding the answer the agent gave and the four
passages that were in the prompt when it gave it. Nothing here is
filled in automatically: the marks below are what the C3 entry in
thesis/measurements.md counts.

**How to mark one section.** Read the answer first, then the passages.
Put an `x` in exactly one box:

- `grounded` - every factual claim in the answer is supported by a
  sentence of the passages printed under it. Numbers, dates, fees and
  names have to match the passages, not your own knowledge of Serbia.
- `ungrounded` - at least one claim is not in the passages, or
  contradicts them. A figure that is right in the world but absent
  from the passages is ungrounded: it was not retrieved, so the agent
  did not read it there.
- `declined` - the agent refused to answer. Nothing to judge for
  grounding; whether the refusal was correct is C2's question, not
  this one.

Add a line under **Notes** when the verdict needed a decision, so the
entry can say why. Then run:

```
.venv/Scripts/python.exe -m scripts.tally_grounding
```

which counts the marks, refuses a section that is unmarked or marked
twice, and compares your `declined` marks with the ones the automatic
detector found.

---

## 1. [en] just landed in belgrade. do i have to register my address somewhere or does the landlord do it?

- **Answer should come from:** 01_white_card_registration.txt
- **Expected fact:** Within 24 hours; the landlord or accommodation provider is legally obliged
- **Retrieved:** 01_white_card_registration.txt, 01_white_card_registration.txt, 07_visa_regime_entry.txt, 02_temporary_residence_permit.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

If you are staying at a hotel or hostel, the establishment will handle registration automatically. If you are staying with a private host (landlord, friends, family), the landlord must register you at the nearest police station or via the eUprava portal within 24 hours of your arrival.

### The passages the prompt carried

**1. 01_white_card_registration.txt** (from character 2070)

```
ELECTRONIC REGISTRATION
The landlord can also register the foreigner electronically:
- Through the eUprava portal (euprava.gov.rs) using an eID account
- Through the Welcome to Serbia portal (welcometoserbia.gov.rs) for temporary residence applications
- Accommodation facilities use the eTourist system

IMPORTANT NOTES
- Registration must be completed within 24 hours of the foreigner's arrival.
- If you change your address within Serbia, you must re-register at the new address.
- The White Card is required when applying for a temporary residence permit.
- Without the White Card, a foreigner is considered unregistered, which may result in fines or difficulties with visa extensions.
- If staying at a hotel or hostel, the establishment handles registration automatically.
- The White Card does not grant any residence rights by itself — it is simply proof that your address has been reported to the police.
```

**2. 01_white_card_registration.txt** (from character 1333)

```
HOW TO REGISTER — STEP BY STEP
1. The landlord (property owner) or accommodation provider must go to the nearest police station (policijska stanica) in the territory where the accommodation is located.
2. Required documents:
   - Foreigner's passport (original for inspection, photocopy to submit)
   - Landlord's personal ID card (lična karta)
   - Proof of property ownership or rental agreement
   - Landlord's written statement (izjava stanodavca) consenting to register the foreigner at their address
3. The police station issues the White Card (Beli karton) — a registration slip confirming the foreigner's registered address.
4. Keep this document safe — you will need it for virtually every subsequent administrative procedure.
```

**3. 07_visa_regime_entry.txt** (from character 4122)

```
REGISTRATION UPON ARRIVAL
After entering Serbia, every foreigner must register their address within 24 hours:
- Hotels and accommodation facilities register automatically via the eTourist system
- Private hosts (landlords, friends, family) must register the foreigner at the nearest police station or via the eUprava portal
- This produces the White Card (Beli karton) — see the separate document on White Card registration

SPECIAL CATEGORIES

Digital Nomads:
Serbia has become a popular destination for digital nomads due to affordable living costs, good internet infrastructure, and a welcoming attitude toward remote workers. While Serbia does not yet have a specific "digital nomad visa," foreigners can:
- Stay visa-free for up to 90 days
- Apply for temporary residence on various grounds
- Register as freelancers for tax purposes through the frilenseri.purs.gov.rs portal
```

**4. 02_temporary_residence_permit.txt** (from character 1351)

```
HOW TO APPLY — STEP BY STEP
1. Register your address in Serbia (obtain the White Card / Beli karton).
2. Create an account on the Welcome to Serbia portal (welcometoserbia.gov.rs) or the eID portal (eid.gov.rs).
3. Submit the application electronically through the portal, or in person at the competent police department (MUP) responsible for foreigner affairs in your area.
4. Pay the required fees.
5. Wait for processing — the standard government processing time is 19 days for the Single Permit (significantly reduced from the previous 70 days).
6. If approved, visit the competent authority in person to provide biometric data (photo, fingerprints).
7. Receive your biometric residence permit card.
```

---

## 2. [ru] хозяин квартиры говорит что ему надо что-то нести в полицию для моей регистрации - что именно?

- **Answer should come from:** 01_white_card_registration.txt
- **Expected fact:** Паспорт иностранца; личная карта арендодателя; договор аренды или подтверждение собственности; изява станодавца
- **Retrieved:** 01_white_card_registration.txt, 01_white_card_registration.txt, 01_white_card_registration.txt, 01_white_card_registration.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Хозяин квартиры должен принести в полицию следующие документы для вашей регистрации:

1. Паспорт иностранца (оригинал для проверки, photocopy для подачи)
2. Личная карта хозяина (лична карта)
3. Доказательство права собственности на недвижимость или договор аренды
4. Письменное заявление хозяина (изява станодавца) с согласием зарегистрировать иностранца по его адресу.

### The passages the prompt carried

**1. 01_white_card_registration.txt** (from character 1333)

```
HOW TO REGISTER — STEP BY STEP
1. The landlord (property owner) or accommodation provider must go to the nearest police station (policijska stanica) in the territory where the accommodation is located.
2. Required documents:
   - Foreigner's passport (original for inspection, photocopy to submit)
   - Landlord's personal ID card (lična karta)
   - Proof of property ownership or rental agreement
   - Landlord's written statement (izjava stanodavca) consenting to register the foreigner at their address
3. The police station issues the White Card (Beli karton) — a registration slip confirming the foreigner's registered address.
4. Keep this document safe — you will need it for virtually every subsequent administrative procedure.
```

**2. 01_white_card_registration.txt** (from character 2070)

```
ELECTRONIC REGISTRATION
The landlord can also register the foreigner electronically:
- Through the eUprava portal (euprava.gov.rs) using an eID account
- Through the Welcome to Serbia portal (welcometoserbia.gov.rs) for temporary residence applications
- Accommodation facilities use the eTourist system

IMPORTANT NOTES
- Registration must be completed within 24 hours of the foreigner's arrival.
- If you change your address within Serbia, you must re-register at the new address.
- The White Card is required when applying for a temporary residence permit.
- Without the White Card, a foreigner is considered unregistered, which may result in fines or difficulties with visa extensions.
- If staying at a hotel or hostel, the establishment handles registration automatically.
- The White Card does not grant any residence rights by itself — it is simply proof that your address has been reported to the police.
```

**3. 01_white_card_registration.txt** (from character 776)

```
WHO IS RESPONSIBLE FOR REGISTRATION?
- Hotels, hostels, boarding houses, and other accommodation facilities register the foreigner's stay electronically through the eTourist application.
- If you are staying in private accommodation (rented apartment, staying with friends or family), the landlord or the person providing accommodation is legally obligated to register the foreigner's stay with the competent authority within 24 hours.
- The foreigner's stay can also be registered electronically on the eAdministration (eUprava) portal at euprava.gov.rs.
```

**4. 01_white_card_registration.txt** (from character 0)

```
WHITE CARD (BELI KARTON) — REGISTRATION OF FOREIGNER'S ADDRESS IN SERBIA

WHAT IS THE WHITE CARD?
The White Card (Beli karton) is the mandatory registration of a foreigner's residential address in Serbia. Every foreigner staying in Serbia must register their address with the police within 24 hours of arrival. This is the first and most important step for any foreigner entering Serbia. Without it, you cannot proceed with any other administrative procedures such as obtaining a temporary residence permit, opening a bank account, or registering for health insurance.

WHO MUST REGISTER?
All foreign nationals staying in the Republic of Serbia, regardless of the purpose or duration of their stay. This applies to tourists, business visitors, and long-term residents alike.
```

---

## 3. [sr] zasto svi traze beli karton pre bilo cega drugog?

- **Answer should come from:** 01_white_card_registration.txt
- **Expected fact:** Bez njega se ne može podneti zahtev za boravak; otvoriti bankovni račun ni prijaviti se za zdravstveno osiguranje
- **Retrieved:** 01_white_card_registration.txt, 06_utility_bills_payments.txt, 04_health_insurance.txt, 01_white_card_registration.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Beli karton je obavezan za registraciju adrese stranog državljanina u Srbiji. Bez njega, ne možete nastaviti sa drugim administrativnim procedurama kao što su dobijanje privremene dozvole boravka, otvaranje bankovnog računa ili registracija za zdravstveno osiguranje. To je prvi i najvažniji korak za svakog stranca koji ulazi u Srbiju.

### The passages the prompt carried

**1. 01_white_card_registration.txt** (from character 0)

```
WHITE CARD (BELI KARTON) — REGISTRATION OF FOREIGNER'S ADDRESS IN SERBIA

WHAT IS THE WHITE CARD?
The White Card (Beli karton) is the mandatory registration of a foreigner's residential address in Serbia. Every foreigner staying in Serbia must register their address with the police within 24 hours of arrival. This is the first and most important step for any foreigner entering Serbia. Without it, you cannot proceed with any other administrative procedures such as obtaining a temporary residence permit, opening a bank account, or registering for health insurance.

WHO MUST REGISTER?
All foreign nationals staying in the Republic of Serbia, regardless of the purpose or duration of their stay. This applies to tourists, business visitors, and long-term residents alike.
```

**2. 06_utility_bills_payments.txt** (from character 2937)

```
SIM card: You can purchase a SIM card without documents, but activation and full use require a valid passport. Visit the operator's office to link the card to your passport.

BUILDING MAINTENANCE (ODRŽAVANJE ZGRADE)
- A separate fee for common areas, elevator maintenance, and building cleaning
- Charged per square meter of your apartment
- Usually collected through the Infostan/Informatika bill or separately by the building management (skupština stanara)

HOW TO PAY UTILITY BILLS

Method 1: Post Office (Pošta)
- Bring the paper bill (nalog za uplatu) to any post office
- Pay in cash (RSD only)
- Lowest commission fees (starting from 65 RSD per payment)
- Available at all post offices across Serbia

Method 2: Exchange Offices (Menjačnica)
- Many exchange offices (such as Exclusive Change) accept bill payments
- Commission: typically 0.60%, minimum 65 RSD
- Available evenings and weekends at many locations
```

**3. 04_health_insurance.txt** (from character 2367)

```
HEALTH INSURANCE CARD (ZDRAVSTVENA KARTICA)
- The card proves your insurance status when visiting public healthcare facilities.
- It must be renewed periodically — check online if automatic renewal has occurred.
- If the card is not renewed automatically, there may be a tax debt. Pay off the debt, obtain a certificate of no debt from the tax office, and take it to RFZO to renew.

IMPORTANT BENEFIT: If a public clinic offers a doctor's appointment with a waiting period of more than 30 days, you can request written confirmation. With this document, you can see a private specialist and submit the receipt to RFZO for reimbursement.

EMERGENCY MEDICAL CARE FOR FOREIGNERS
All foreign citizens in Serbia have the right to emergency medical assistance regardless of insurance status. However, payment depends on bilateral agreements:
```

**4. 01_white_card_registration.txt** (from character 1333)

```
HOW TO REGISTER — STEP BY STEP
1. The landlord (property owner) or accommodation provider must go to the nearest police station (policijska stanica) in the territory where the accommodation is located.
2. Required documents:
   - Foreigner's passport (original for inspection, photocopy to submit)
   - Landlord's personal ID card (lična karta)
   - Proof of property ownership or rental agreement
   - Landlord's written statement (izjava stanodavca) consenting to register the foreigner at their address
3. The police station issues the White Card (Beli karton) — a registration slip confirming the foreigner's registered address.
4. Keep this document safe — you will need it for virtually every subsequent administrative procedure.
```

---

## 4. [en] i got the white card - does that mean im allowed to stay here now?

- **Answer should come from:** 01_white_card_registration.txt
- **Expected fact:** No - it only proves the address was reported to the police
- **Retrieved:** 01_white_card_registration.txt, 01_white_card_registration.txt, 01_white_card_registration.txt, 02_temporary_residence_permit.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

The White Card (Beli karton) is proof that your address has been registered with the police, but it does not grant any residence rights by itself. You must stay in Serbia in accordance with the purpose for which your residence was granted.

### The passages the prompt carried

**1. 01_white_card_registration.txt** (from character 0)

```
WHITE CARD (BELI KARTON) — REGISTRATION OF FOREIGNER'S ADDRESS IN SERBIA

WHAT IS THE WHITE CARD?
The White Card (Beli karton) is the mandatory registration of a foreigner's residential address in Serbia. Every foreigner staying in Serbia must register their address with the police within 24 hours of arrival. This is the first and most important step for any foreigner entering Serbia. Without it, you cannot proceed with any other administrative procedures such as obtaining a temporary residence permit, opening a bank account, or registering for health insurance.

WHO MUST REGISTER?
All foreign nationals staying in the Republic of Serbia, regardless of the purpose or duration of their stay. This applies to tourists, business visitors, and long-term residents alike.
```

**2. 01_white_card_registration.txt** (from character 2070)

```
ELECTRONIC REGISTRATION
The landlord can also register the foreigner electronically:
- Through the eUprava portal (euprava.gov.rs) using an eID account
- Through the Welcome to Serbia portal (welcometoserbia.gov.rs) for temporary residence applications
- Accommodation facilities use the eTourist system

IMPORTANT NOTES
- Registration must be completed within 24 hours of the foreigner's arrival.
- If you change your address within Serbia, you must re-register at the new address.
- The White Card is required when applying for a temporary residence permit.
- Without the White Card, a foreigner is considered unregistered, which may result in fines or difficulties with visa extensions.
- If staying at a hotel or hostel, the establishment handles registration automatically.
- The White Card does not grant any residence rights by itself — it is simply proof that your address has been reported to the police.
```

**3. 01_white_card_registration.txt** (from character 1333)

```
HOW TO REGISTER — STEP BY STEP
1. The landlord (property owner) or accommodation provider must go to the nearest police station (policijska stanica) in the territory where the accommodation is located.
2. Required documents:
   - Foreigner's passport (original for inspection, photocopy to submit)
   - Landlord's personal ID card (lična karta)
   - Proof of property ownership or rental agreement
   - Landlord's written statement (izjava stanodavca) consenting to register the foreigner at their address
3. The police station issues the White Card (Beli karton) — a registration slip confirming the foreigner's registered address.
4. Keep this document safe — you will need it for virtually every subsequent administrative procedure.
```

**4. 02_temporary_residence_permit.txt** (from character 5055)

```
IMPORTANT NOTES
- You must stay in Serbia in accordance with the purpose for which residence was granted.
- If you entered with a Visa D, submit the residence application before the visa expires.
- Minor foreigners (under 18) must apply in person with a parent or guardian.
- All notifications regarding your application are delivered through the eDatabox if you applied online.
- The biometric residence permit card also serves as an identity document in Serbia.

AFTER 3 YEARS — PERMANENT RESIDENCE
After 3 continuous years of temporary residence, you may be eligible to apply for permanent residence (stalni boravak), which is granted indefinitely and issued as a foreigner ID card valid for 5 years (renewable).

LEGAL BASIS
Law on Foreigners ("Official Gazette of RS" No. 24/2018, 31/2019, and 62/2023)
Law on Employment of Foreigners
```

---

## 5. [en] how long is the wait for the single permit now? heard it used to take forever

- **Answer should come from:** 02_temporary_residence_permit.txt
- **Expected fact:** 19 days; down from 70
- **Retrieved:** 02_temporary_residence_permit.txt, 02_temporary_residence_permit.txt, 02_temporary_residence_permit.txt, 08_egovernment_digital_services.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

The standard processing time for the Single Permit is now 19 days, significantly reduced from the previous 70 days.

### The passages the prompt carried

**1. 02_temporary_residence_permit.txt** (from character 4124)

```
SINGLE PERMIT (UNIFIED PERMIT FOR RESIDENCE AND WORK)
Since February 2024, foreigners employed in Serbia receive a Single Permit that combines both residence and work authorization. Key points:
- Applied for through the Welcome to Serbia portal
- Standard processing time: 19 days
- Visa-free nationals must apply for a Single Permit (Visa D is no longer available for most visa-free nationals who want to work)
- The employer can submit the application on behalf of the employee with a power of attorney
- Changes of employer or basis of employment can be submitted through the portal

CERTIFICATE OF INITIATED PROCEDURE
After submitting your application, you receive a certificate confirming that the procedure has been initiated. This certificate includes your foreigner registration number (EBS), which you need for exercising rights and obligations in Serbia (opening a bank account, registering for health insurance, etc.).
```

**2. 02_temporary_residence_permit.txt** (from character 1351)

```
HOW TO APPLY — STEP BY STEP
1. Register your address in Serbia (obtain the White Card / Beli karton).
2. Create an account on the Welcome to Serbia portal (welcometoserbia.gov.rs) or the eID portal (eid.gov.rs).
3. Submit the application electronically through the portal, or in person at the competent police department (MUP) responsible for foreigner affairs in your area.
4. Pay the required fees.
5. Wait for processing — the standard government processing time is 19 days for the Single Permit (significantly reduced from the previous 70 days).
6. If approved, visit the competent authority in person to provide biometric data (photo, fingerprints).
7. Receive your biometric residence permit card.
```

**3. 02_temporary_residence_permit.txt** (from character 0)

```
TEMPORARY RESIDENCE PERMIT (PRIVREMENI BORAVAK) IN SERBIA

WHAT IS A TEMPORARY RESIDENCE PERMIT?
A temporary residence permit is an official authorization that allows a foreign citizen to legally reside in the Republic of Serbia for a period longer than 90 days. Since February 2024, Serbia introduced the Single Permit (Jedinstvena dozvola), which combines both residence and work authorization in one document. The permit is issued as a biometric ID card.

WHO NEEDS A TEMPORARY RESIDENCE PERMIT?
Any foreign national who plans to stay in Serbia for more than 90 days within a 180-day period must apply for a temporary residence permit. Citizens of visa-free countries can stay up to 90 days without a permit, but must apply before this period expires if they wish to remain longer.
```

**4. 08_egovernment_digital_services.txt** (from character 2555)

```
eFOREIGNER PORTAL
Website: eforeigner.welcometoserbia.gov.rs

Specifically for temporary residence and Single Permit applications:
- Submit residence applications electronically
- Submit changes of employer or employment basis
- Track application progress
- Receive decisions through the eDatabox
- Access help and FAQs

Important notes:
- Applications for minors (under 18) cannot be submitted online — must be done in person
- If the system asks for a Visa ID during extension and you do not have one, contact support at office@ite.gov.rs to have your account reset with your EBS number
- Once submitted, applications cannot be cancelled through the portal

ePOREZI PORTAL (eTAXES)
Website: eporezi.purs.gov.rs

The Tax Administration's electronic services portal:
- Submit tax returns electronically
- Monitor submitted returns
- View taxpayer account balances
- Grant/revoke authorization for tax representatives
- Available daily 6:00 AM to midnight
- Test environment available for practice
```

---

## 6. [en] im married to a serbian citizen. how much does the residence permit cost me?

- **Answer should come from:** 02_temporary_residence_permit.txt
- **Expected fact:** 11350 RSD
- **Retrieved:** 02_temporary_residence_permit.txt, 02_temporary_residence_permit.txt, 02_temporary_residence_permit.txt, 02_temporary_residence_permit.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

The fee for residence based on marriage with a Serbian citizen is 11,350 RSD.

### The passages the prompt carried

**1. 02_temporary_residence_permit.txt** (from character 3370)

```
FEES
Fees vary depending on the basis for residence. Examples:
- Standard temporary residence fee: approximately 20,660 RSD (as of 2024)
- Fee for residence based on marriage with a Serbian citizen: 11,350 RSD
- Biometric card issuance fee: additional charge applies
- Fees are paid via the eUprava portal or at a bank/post office

DURATION AND RENEWAL
- Temporary residence can be granted for up to 3 years and can be extended for the same period.
- The renewal application should be submitted at the earliest 60 days before the current permit expires.
- You must continue to meet the conditions for the approved basis of residence throughout its duration.
- If circumstances change (e.g., change of employer), you must notify the competent authority.
```

**2. 02_temporary_residence_permit.txt** (from character 1351)

```
HOW TO APPLY — STEP BY STEP
1. Register your address in Serbia (obtain the White Card / Beli karton).
2. Create an account on the Welcome to Serbia portal (welcometoserbia.gov.rs) or the eID portal (eid.gov.rs).
3. Submit the application electronically through the portal, or in person at the competent police department (MUP) responsible for foreigner affairs in your area.
4. Pay the required fees.
5. Wait for processing — the standard government processing time is 19 days for the Single Permit (significantly reduced from the previous 70 days).
6. If approved, visit the competent authority in person to provide biometric data (photo, fingerprints).
7. Receive your biometric residence permit card.
```

**3. 02_temporary_residence_permit.txt** (from character 0)

```
TEMPORARY RESIDENCE PERMIT (PRIVREMENI BORAVAK) IN SERBIA

WHAT IS A TEMPORARY RESIDENCE PERMIT?
A temporary residence permit is an official authorization that allows a foreign citizen to legally reside in the Republic of Serbia for a period longer than 90 days. Since February 2024, Serbia introduced the Single Permit (Jedinstvena dozvola), which combines both residence and work authorization in one document. The permit is issued as a biometric ID card.

WHO NEEDS A TEMPORARY RESIDENCE PERMIT?
Any foreign national who plans to stay in Serbia for more than 90 days within a 180-day period must apply for a temporary residence permit. Citizens of visa-free countries can stay up to 90 days without a permit, but must apply before this period expires if they wish to remain longer.
```

**4. 02_temporary_residence_permit.txt** (from character 4124)

```
SINGLE PERMIT (UNIFIED PERMIT FOR RESIDENCE AND WORK)
Since February 2024, foreigners employed in Serbia receive a Single Permit that combines both residence and work authorization. Key points:
- Applied for through the Welcome to Serbia portal
- Standard processing time: 19 days
- Visa-free nationals must apply for a Single Permit (Visa D is no longer available for most visa-free nationals who want to work)
- The employer can submit the application on behalf of the employee with a power of attorney
- Changes of employer or basis of employment can be submitted through the portal

CERTIFICATE OF INITIATED PROCEDURE
After submitting your application, you receive a certificate confirming that the procedure has been initiated. This certificate includes your foreigner registration number (EBS), which you need for exercising rights and obligations in Serbia (opening a bank account, registering for health insurance, etc.).
```

---

## 7. [ru] боравак заканчивается через пару месяцев - когда уже можно подавать на продление?

- **Answer should come from:** 02_temporary_residence_permit.txt
- **Expected fact:** Не ранее чем за 60 дней до истечения
- **Retrieved:** 02_temporary_residence_permit.txt, 02_temporary_residence_permit.txt, 07_visa_regime_entry.txt, 07_visa_regime_entry.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Вы можете подавать на продление не ранее чем за 60 дней до истечения текущего разрешения на временный боравак.

### The passages the prompt carried

**1. 02_temporary_residence_permit.txt** (from character 3370)

```
FEES
Fees vary depending on the basis for residence. Examples:
- Standard temporary residence fee: approximately 20,660 RSD (as of 2024)
- Fee for residence based on marriage with a Serbian citizen: 11,350 RSD
- Biometric card issuance fee: additional charge applies
- Fees are paid via the eUprava portal or at a bank/post office

DURATION AND RENEWAL
- Temporary residence can be granted for up to 3 years and can be extended for the same period.
- The renewal application should be submitted at the earliest 60 days before the current permit expires.
- You must continue to meet the conditions for the approved basis of residence throughout its duration.
- If circumstances change (e.g., change of employer), you must notify the competent authority.
```

**2. 02_temporary_residence_permit.txt** (from character 5055)

```
IMPORTANT NOTES
- You must stay in Serbia in accordance with the purpose for which residence was granted.
- If you entered with a Visa D, submit the residence application before the visa expires.
- Minor foreigners (under 18) must apply in person with a parent or guardian.
- All notifications regarding your application are delivered through the eDatabox if you applied online.
- The biometric residence permit card also serves as an identity document in Serbia.

AFTER 3 YEARS — PERMANENT RESIDENCE
After 3 continuous years of temporary residence, you may be eligible to apply for permanent residence (stalni boravak), which is granted indefinitely and issued as a foreigner ID card valid for 5 years (renewable).

LEGAL BASIS
Law on Foreigners ("Official Gazette of RS" No. 24/2018, 31/2019, and 62/2023)
Law on Employment of Foreigners
```

**3. 07_visa_regime_entry.txt** (from character 938)

```
Important: The 90/180 rule means you can stay a maximum of 90 days in any rolling 180-day period. This is not 90 consecutive days — it is cumulative. If you stay 90 days and leave, you cannot re-enter until 90 more days have passed.

VISA RUN
A "visa run" is the common practice of leaving Serbia before the 90-day limit and re-entering to reset the stay period. Common visa run destinations from Serbia include:
- Bosnia and Herzegovina (closest border crossings)
- North Macedonia
- Hungary
- Romania
- Bulgaria

Procedure: Drive or take transport to the border, exit Serbia (get an exit stamp), enter the neighboring country briefly, then re-enter Serbia (get a new entry stamp). Some organized visa runs involve walking across a border checkpoint and returning the same day.
```

**4. 07_visa_regime_entry.txt** (from character 1718)

```
Warning: Serbian border authorities are aware of this practice. While it is technically legal if done within the 90/180 rule, repeatedly doing so without legitimate purpose may raise questions. If you plan to stay long-term, applying for a temporary residence permit is strongly recommended.

VISA TYPES

Type C — Short-Stay Visa:
- For stays up to 90 days
- For tourism, business visits, family visits, medical treatment, etc.
- Required for nationals of countries that do not have visa-free agreements with Serbia
- Applied for at Serbian embassies/consulates abroad or online via welcometoserbia.gov.rs
```

---

## 8. [sr] predao sam zahtev za boravak i dobio neku potvrdu sa brojem - sta mi to znaci?

- **Answer should come from:** 02_temporary_residence_permit.txt
- **Expected fact:** Potvrda o pokrenutom postupku; sadrži EBS broj stranca
- **Retrieved:** 01_white_card_registration.txt, 01_white_card_registration.txt, 01_white_card_registration.txt, 02_temporary_residence_permit.txt
- **Document retrieved:** yes at position 4
- **Reads as a decline:** no

**Verdict:** [ ] grounded  [x] ungrounded  [ ] declined

**Notes:** The certificate chunk (potvrda o pokrenutom postupku, EBS) was not among the four passages. Only the eDatabox sentence comes from them; that the certificate proves submission and can be used to track status is the agent's own addition.

### The answer

To znači da ste predali zahtev za boravak i dobili potvrdu koja sadrži broj vašeg zahteva. Ova potvrda je važna jer služi kao dokaz da ste podneli zahtev i može se koristiti za praćenje statusa vašeg zahteva. Svi obaveštenja u vezi sa vašim zahtevom biće dostavljena putem eDatabox-a ako ste se prijavili online.

### The passages the prompt carried

**1. 01_white_card_registration.txt** (from character 1333)

```
HOW TO REGISTER — STEP BY STEP
1. The landlord (property owner) or accommodation provider must go to the nearest police station (policijska stanica) in the territory where the accommodation is located.
2. Required documents:
   - Foreigner's passport (original for inspection, photocopy to submit)
   - Landlord's personal ID card (lična karta)
   - Proof of property ownership or rental agreement
   - Landlord's written statement (izjava stanodavca) consenting to register the foreigner at their address
3. The police station issues the White Card (Beli karton) — a registration slip confirming the foreigner's registered address.
4. Keep this document safe — you will need it for virtually every subsequent administrative procedure.
```

**2. 01_white_card_registration.txt** (from character 2070)

```
ELECTRONIC REGISTRATION
The landlord can also register the foreigner electronically:
- Through the eUprava portal (euprava.gov.rs) using an eID account
- Through the Welcome to Serbia portal (welcometoserbia.gov.rs) for temporary residence applications
- Accommodation facilities use the eTourist system

IMPORTANT NOTES
- Registration must be completed within 24 hours of the foreigner's arrival.
- If you change your address within Serbia, you must re-register at the new address.
- The White Card is required when applying for a temporary residence permit.
- Without the White Card, a foreigner is considered unregistered, which may result in fines or difficulties with visa extensions.
- If staying at a hotel or hostel, the establishment handles registration automatically.
- The White Card does not grant any residence rights by itself — it is simply proof that your address has been reported to the police.
```

**3. 01_white_card_registration.txt** (from character 776)

```
WHO IS RESPONSIBLE FOR REGISTRATION?
- Hotels, hostels, boarding houses, and other accommodation facilities register the foreigner's stay electronically through the eTourist application.
- If you are staying in private accommodation (rented apartment, staying with friends or family), the landlord or the person providing accommodation is legally obligated to register the foreigner's stay with the competent authority within 24 hours.
- The foreigner's stay can also be registered electronically on the eAdministration (eUprava) portal at euprava.gov.rs.
```

**4. 02_temporary_residence_permit.txt** (from character 5055)

```
IMPORTANT NOTES
- You must stay in Serbia in accordance with the purpose for which residence was granted.
- If you entered with a Visa D, submit the residence application before the visa expires.
- Minor foreigners (under 18) must apply in person with a parent or guardian.
- All notifications regarding your application are delivered through the eDatabox if you applied online.
- The biometric residence permit card also serves as an identity document in Serbia.

AFTER 3 YEARS — PERMANENT RESIDENCE
After 3 continuous years of temporary residence, you may be eligible to apply for permanent residence (stalni boravak), which is granted indefinitely and issued as a foreigner ID card valid for 5 years (renewable).

LEGAL BASIS
Law on Foreigners ("Official Gazette of RS" No. 24/2018, 31/2019, and 62/2023)
Law on Employment of Foreigners
```

---

## 9. [en] ive been here almost a year now. does anything change with my bank account after that?

- **Answer should come from:** 03_opening_bank_account.txt
- **Expected fact:** After one year; the bank closes the non-resident account and opens a resident one
- **Retrieved:** 03_opening_bank_account.txt, 03_opening_bank_account.txt, 03_opening_bank_account.txt, 03_opening_bank_account.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Yes, after one year, you acquire resident status and must notify the bank. The bank will close your non-resident account and open a resident account.

### The passages the prompt carried

**1. 03_opening_bank_account.txt** (from character 808)

```
After one year, you acquire resident status and must notify the bank. The bank will close your non-resident account and open a resident account.

OPENING A NON-RESIDENT PERSONAL ACCOUNT

Required documents:
- Valid passport (original for verification, photocopy for the bank)
- Proof of registered address in Serbia (White Card / Beli karton or rental agreement)
- Completed bank application forms (provided by the bank)

The process:
1. Visit a bank branch in person. At present, fully remote account opening by the individual is not possible at most banks, though some allow an authorized representative (attorney with power of attorney) to complete the process on your behalf.
2. The bank conducts KYC (Know Your Customer) verification.
3. Processing time varies: some banks complete the procedure the same day, others may take several days.
4. Once approved, you receive account details, a debit card (Visa, MasterCard, or DinaCard), and access to online/mobile banking.
```

**2. 03_opening_bank_account.txt** (from character 1784)

```
OPENING A RESIDENT PERSONAL ACCOUNT

Once you have been in Serbia for more than one year with a valid residence permit, you transition to resident status. The process is similar to opening a non-resident account, but with additional documentation:
- Valid passport
- Temporary or permanent residence permit (biometric card)
- Proof of address
- Tax Identification Number (PIB) — may be required by some banks

POPULAR BANKS FOR FOREIGNERS

Raiffeisen Banka:
- Part of Raiffeisen Bank International group
- Accounts in RSD and multiple foreign currencies
- Online banking available in Serbian and English
- No minimum balance requirement
- Account opening may take 2 to 4 weeks
- Account signatory must visit Serbia in person
```

**3. 03_opening_bank_account.txt** (from character 0)

```
OPENING A BANK ACCOUNT IN SERBIA AS A FOREIGNER

OVERVIEW
Foreign citizens can open bank accounts in Serbia, both for Serbian dinars (RSD) and foreign currencies (EUR, USD, CHF, GBP, and others). The process differs depending on whether you are classified as a resident or non-resident. As of January 2025, there are 20 licensed commercial banks operating in Serbia, including local, EU, Russian, American, Turkish, UAE, and Chinese banks.

RESIDENT VS NON-RESIDENT STATUS
Under the Law on Foreign Exchange Operations ("Official Gazette of RS" Nos. 62/2006, 31/2011, 119/2012, 139/2014, and 30/2018):
- Non-resident: A foreign citizen in the first year of their stay in Serbia.
- Resident: A foreign citizen who, on the basis of a residence permit or work visa, has stayed in Serbia for more than one year.

After one year, you acquire resident status and must notify the bank. The bank will close your non-resident account and open a resident account.

OPENING A NON-RESIDENT PERSONAL ACCOUNT
```

**4. 03_opening_bank_account.txt** (from character 2976)

```
Other banks: Erste Bank, UniCredit (now OTP), Mobi Banka (fully mobile banking), AIK Banka, NLB Banka, Poštanska Štedionica

FEES AND COSTS
- Account opening: Usually free of charge (some premium banks like Raiffeisen may charge up to EUR 1,000 for corporate accounts)
- Monthly maintenance: Typically 5-10 EUR for RSD accounts; foreign currency account maintenance is often free
- International transfers: Fees vary by bank — compare before choosing
- Card issuance: Usually included in the account package
- Serbia is becoming part of SEPA from 2025, which will simplify EUR transfers

BANKING SERVICES AVAILABLE
- Online banking (most banks offer English language support)
- Mobile banking applications
- Debit cards (Visa, MasterCard, DinaCard accepted locally)
- International SWIFT transfers
- Standing orders for automatic bill payments
- Multi-currency accounts (RSD, EUR, USD, CHF, GBP, etc.)
```

---

## 10. [ru] что взять с собой в банк чтобы открыть счет? я тут недавно

- **Answer should come from:** 03_opening_bank_account.txt
- **Expected fact:** Паспорт; подтверждение адреса (белый картон или договор аренды); банковские анкеты
- **Retrieved:** 03_opening_bank_account.txt, 03_opening_bank_account.txt, 03_opening_bank_account.txt, 03_opening_bank_account.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Чтобы открыть счет в банке как нерезидент, вам понадобятся следующие документы:
- Действующий паспорт (оригинал для проверки, photocopy для банка)
- Доказательство зарегистрированного адреса в Сербии (Белая карта / Beli karton или договор аренды)
- Заполненные банковские формы заявки (предоставляются банком)

### The passages the prompt carried

**1. 03_opening_bank_account.txt** (from character 808)

```
After one year, you acquire resident status and must notify the bank. The bank will close your non-resident account and open a resident account.

OPENING A NON-RESIDENT PERSONAL ACCOUNT

Required documents:
- Valid passport (original for verification, photocopy for the bank)
- Proof of registered address in Serbia (White Card / Beli karton or rental agreement)
- Completed bank application forms (provided by the bank)

The process:
1. Visit a bank branch in person. At present, fully remote account opening by the individual is not possible at most banks, though some allow an authorized representative (attorney with power of attorney) to complete the process on your behalf.
2. The bank conducts KYC (Know Your Customer) verification.
3. Processing time varies: some banks complete the procedure the same day, others may take several days.
4. Once approved, you receive account details, a debit card (Visa, MasterCard, or DinaCard), and access to online/mobile banking.
```

**2. 03_opening_bank_account.txt** (from character 1784)

```
OPENING A RESIDENT PERSONAL ACCOUNT

Once you have been in Serbia for more than one year with a valid residence permit, you transition to resident status. The process is similar to opening a non-resident account, but with additional documentation:
- Valid passport
- Temporary or permanent residence permit (biometric card)
- Proof of address
- Tax Identification Number (PIB) — may be required by some banks

POPULAR BANKS FOR FOREIGNERS

Raiffeisen Banka:
- Part of Raiffeisen Bank International group
- Accounts in RSD and multiple foreign currencies
- Online banking available in Serbian and English
- No minimum balance requirement
- Account opening may take 2 to 4 weeks
- Account signatory must visit Serbia in person
```

**3. 03_opening_bank_account.txt** (from character 0)

```
OPENING A BANK ACCOUNT IN SERBIA AS A FOREIGNER

OVERVIEW
Foreign citizens can open bank accounts in Serbia, both for Serbian dinars (RSD) and foreign currencies (EUR, USD, CHF, GBP, and others). The process differs depending on whether you are classified as a resident or non-resident. As of January 2025, there are 20 licensed commercial banks operating in Serbia, including local, EU, Russian, American, Turkish, UAE, and Chinese banks.

RESIDENT VS NON-RESIDENT STATUS
Under the Law on Foreign Exchange Operations ("Official Gazette of RS" Nos. 62/2006, 31/2011, 119/2012, 139/2014, and 30/2018):
- Non-resident: A foreign citizen in the first year of their stay in Serbia.
- Resident: A foreign citizen who, on the basis of a residence permit or work visa, has stayed in Serbia for more than one year.

After one year, you acquire resident status and must notify the bank. The bank will close your non-resident account and open a resident account.

OPENING A NON-RESIDENT PERSONAL ACCOUNT
```

**4. 03_opening_bank_account.txt** (from character 3879)

```
IMPORTANT NOTES
- Banks in Serbia are regulated by the National Bank of Serbia (NBS), established in 1884.
- Deposits in Serbian banks are insured (deposit guarantee scheme).
- Banks must comply with anti-money laundering regulations but cannot refuse service on discriminatory grounds (citizenship, gender, religion). If refused, the bank must provide reasons in writing.
- Some banks have limited reasons for opening non-resident accounts (e.g., specific business purposes).
- A Tax Identification Number (PIB) may be required by certain banks — especially for freelancers or business owners.
- Having a local bank account is essential for: receiving salary, paying utility bills, paying taxes, and daily transactions.
- The IPS (Instant Payment System) enables instant transfers between Serbian bank accounts.
```

---

## 11. [sr] koja banka ima najvise bankomata kod nas?

- **Answer should come from:** 03_opening_bank_account.txt
- **Expected fact:** OTP banka; 294 bankomata i 155 filijala
- **Retrieved:** 03_opening_bank_account.txt, 03_opening_bank_account.txt, 03_opening_bank_account.txt, 03_opening_bank_account.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

OTP Banka ima najviše bankomata u Srbiji, sa 294 bankomata.

### The passages the prompt carried

**1. 03_opening_bank_account.txt** (from character 2510)

```
OTP Banka:
- Largest ATM network in Serbia (294 ATMs, 155 branches)
- Praktik package: RSD and foreign currency accounts, MasterCard Gold debit card, DinaCard
- Monthly maintenance: ~400 RSD (~3 EUR), waived if spending exceeds 15,000 RSD/month
- Klasik package: basic option with MasterCard Standard, ~300 RSD/month maintenance

Banca Intesa:
- One of the largest banks in Serbia
- Full range of personal banking services
- Commission-free bill payments available

Other banks: Erste Bank, UniCredit (now OTP), Mobi Banka (fully mobile banking), AIK Banka, NLB Banka, Poštanska Štedionica
```

**2. 03_opening_bank_account.txt** (from character 2976)

```
Other banks: Erste Bank, UniCredit (now OTP), Mobi Banka (fully mobile banking), AIK Banka, NLB Banka, Poštanska Štedionica

FEES AND COSTS
- Account opening: Usually free of charge (some premium banks like Raiffeisen may charge up to EUR 1,000 for corporate accounts)
- Monthly maintenance: Typically 5-10 EUR for RSD accounts; foreign currency account maintenance is often free
- International transfers: Fees vary by bank — compare before choosing
- Card issuance: Usually included in the account package
- Serbia is becoming part of SEPA from 2025, which will simplify EUR transfers

BANKING SERVICES AVAILABLE
- Online banking (most banks offer English language support)
- Mobile banking applications
- Debit cards (Visa, MasterCard, DinaCard accepted locally)
- International SWIFT transfers
- Standing orders for automatic bill payments
- Multi-currency accounts (RSD, EUR, USD, CHF, GBP, etc.)
```

**3. 03_opening_bank_account.txt** (from character 1784)

```
OPENING A RESIDENT PERSONAL ACCOUNT

Once you have been in Serbia for more than one year with a valid residence permit, you transition to resident status. The process is similar to opening a non-resident account, but with additional documentation:
- Valid passport
- Temporary or permanent residence permit (biometric card)
- Proof of address
- Tax Identification Number (PIB) — may be required by some banks

POPULAR BANKS FOR FOREIGNERS

Raiffeisen Banka:
- Part of Raiffeisen Bank International group
- Accounts in RSD and multiple foreign currencies
- Online banking available in Serbian and English
- No minimum balance requirement
- Account opening may take 2 to 4 weeks
- Account signatory must visit Serbia in person
```

**4. 03_opening_bank_account.txt** (from character 4693)

```
BILL PAYMENTS THROUGH BANK ACCOUNTS
Once you have a bank account, you can set up:
- Standing orders for recurring utility payments (EPS electricity, Infostan, internet)
- e-Banking payments using payment slips (nalog za uplatu)
- IPS QR code payments for instant settlement

LEGAL BASIS
Law on Foreign Exchange Operations ("Official Gazette of RS" Nos. 62/2006, 31/2011, 119/2012, 139/2014, and 30/2018)
Law on Payment Services ("Official Gazette of RS" Nos. 139/2014 and 44/2018)
Law on Prevention of Money Laundering and Terrorism Financing

USEFUL LINKS
- National Bank of Serbia: nbs.rs
- Welcome to Serbia — Banking info: welcometoserbia.gov.rs/opening-bank-accounts-for-residents-and-non-residents

Source: Welcome to Serbia portal (welcometoserbia.gov.rs), National Bank of Serbia (nbs.rs), Wise.com, Zunic Law
```

---

## 12. [en] bank asked me for a pib. is that normal or do they ask everyone?

- **Answer should come from:** 03_opening_bank_account.txt
- **Expected fact:** Freelancers and business owners
- **Retrieved:** 05_tax_registration_pib.txt, 05_tax_registration_pib.txt, 03_opening_bank_account.txt, 03_opening_bank_account.txt
- **Document retrieved:** yes at position 3
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

A Tax Identification Number (PIB) may be required by certain banks, especially for freelancers or business owners. It is not uncommon for banks to ask for a PIB, but it may not be required for everyone.

### The passages the prompt carried

**1. 05_tax_registration_pib.txt** (from character 759)

```
HOW TO OBTAIN A PIB

For individuals (including foreigners):
1. Online: Through the ePorezi (eTaxes) portal at purs.gov.rs
2. In person: At a local Tax Administration office
3. Required documents:
   - Valid passport or ID card
   - Proof of address in Serbia (residence permit or White Card)
   - Foreigner registration number (EBS) if applicable
4. There is no fee for PIB registration.

For businesses:
- The PIB is automatically assigned when registering a company with the Serbian Business Registers Agency (APR — Agencija za privredne registre).
- The PIB also serves as the VAT number for businesses registered for VAT.

TAX ADMINISTRATION (PORESKA UPRAVA)
- The Tax Administration is a body within the Ministry of Finance.
- It carries out assessment, control, and collection of public revenues.
- Head Office: Cara Dušana 145, Belgrade-Zemun
- Website: purs.gov.rs (Serbian and English)
- eTaxes portal: eporezi.purs.gov.rs
```

**2. 05_tax_registration_pib.txt** (from character 0)

```
TAX REGISTRATION AND TAX IDENTIFICATION NUMBER (PIB) IN SERBIA

OVERVIEW
The Tax Identification Number (PIB — Poreski Identifikacioni Broj) is a unique 9-digit number assigned by the Tax Administration of the Republic of Serbia (Poreska uprava) to individuals and legal entities for tax purposes. It is required for virtually all financial and legal transactions in Serbia, including employment, banking, property ownership, and business activity.

WHO NEEDS A PIB?
- All individuals subject to tax obligations in Serbia (employees, freelancers, property owners)
- All legal entities registered in Serbia (companies, sole proprietors, non-profits)
- Foreign citizens working or earning income in Serbia
- Foreign companies with business activities in Serbia

HOW TO OBTAIN A PIB
```

**3. 03_opening_bank_account.txt** (from character 3879)

```
IMPORTANT NOTES
- Banks in Serbia are regulated by the National Bank of Serbia (NBS), established in 1884.
- Deposits in Serbian banks are insured (deposit guarantee scheme).
- Banks must comply with anti-money laundering regulations but cannot refuse service on discriminatory grounds (citizenship, gender, religion). If refused, the bank must provide reasons in writing.
- Some banks have limited reasons for opening non-resident accounts (e.g., specific business purposes).
- A Tax Identification Number (PIB) may be required by certain banks — especially for freelancers or business owners.
- Having a local bank account is essential for: receiving salary, paying utility bills, paying taxes, and daily transactions.
- The IPS (Instant Payment System) enables instant transfers between Serbian bank accounts.
```

**4. 03_opening_bank_account.txt** (from character 4693)

```
BILL PAYMENTS THROUGH BANK ACCOUNTS
Once you have a bank account, you can set up:
- Standing orders for recurring utility payments (EPS electricity, Infostan, internet)
- e-Banking payments using payment slips (nalog za uplatu)
- IPS QR code payments for instant settlement

LEGAL BASIS
Law on Foreign Exchange Operations ("Official Gazette of RS" Nos. 62/2006, 31/2011, 119/2012, 139/2014, and 30/2018)
Law on Payment Services ("Official Gazette of RS" Nos. 139/2014 and 44/2018)
Law on Prevention of Money Laundering and Terrorism Financing

USEFUL LINKS
- National Bank of Serbia: nbs.rs
- Welcome to Serbia — Banking info: welcometoserbia.gov.rs/opening-bank-accounts-for-residents-and-non-residents

Source: Welcome to Serbia portal (welcometoserbia.gov.rs), National Bank of Serbia (nbs.rs), Wise.com, Zunic Law
```

---

## 13. [en] dom zdravlja gave me an appointment in two months. can i just go private and get the money back?

- **Answer should come from:** 04_health_insurance.txt
- **Expected fact:** Ask for written confirmation; see a private specialist and claim reimbursement from RFZO
- **Retrieved:** 04_health_insurance.txt, 04_health_insurance.txt, 04_health_insurance.txt, 04_health_insurance.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Yes, if the waiting period for your appointment at the public clinic is more than 30 days, you can request written confirmation from them. With that document, you can see a private specialist and submit the receipt to RFZO for reimbursement.

### The passages the prompt carried

**1. 04_health_insurance.txt** (from character 2367)

```
HEALTH INSURANCE CARD (ZDRAVSTVENA KARTICA)
- The card proves your insurance status when visiting public healthcare facilities.
- It must be renewed periodically — check online if automatic renewal has occurred.
- If the card is not renewed automatically, there may be a tax debt. Pay off the debt, obtain a certificate of no debt from the tax office, and take it to RFZO to renew.

IMPORTANT BENEFIT: If a public clinic offers a doctor's appointment with a waiting period of more than 30 days, you can request written confirmation. With this document, you can see a private specialist and submit the receipt to RFZO for reimbursement.

EMERGENCY MEDICAL CARE FOR FOREIGNERS
All foreign citizens in Serbia have the right to emergency medical assistance regardless of insurance status. However, payment depends on bilateral agreements:
```

**2. 04_health_insurance.txt** (from character 925)

```
Public health insurance covers:
- Doctor visits at public health centers (dom zdravlja)
- Hospital treatment and surgeries
- Specialist consultations (with referral from a general practitioner)
- Prescription medications (some are free at pharmacies)
- Laboratory tests and diagnostics
- Maternity care

HOW TO REGISTER FOR PUBLIC HEALTH INSURANCE (OMS)
The application process involves several steps:

1. Determine the submission location based on your status:
   - Employees: submit at the RFZO branch at the location of your employer's headquarters
   - Sole proprietors: submit at the branch corresponding to your business's registered address
   - Unemployed and other categories: submit at your place of residence
```

**3. 04_health_insurance.txt** (from character 3660)

```
Countries using insurance certificates:
- Poland and Great Britain — based on insurance certificates issued in their home country

Countries without bilateral agreements:
- Citizens from countries with no agreement pay for emergency medical services out of pocket during their stay. Upon return to their home country, they may seek reimbursement from their insurance company.

PRIVATE HEALTH INSURANCE
Private health insurance is an alternative or supplement to the public system.

Advantages:
- Shorter waiting times for specialist appointments
- Access to private hospitals and clinics
- Broader choice of doctors and facilities
- Often required for temporary residence applications if you are not employed

Leading private insurance providers in Serbia:
- Generali
- Uniqa
- Wiener Städtische (Vienna Insurance Group)
- Dunav osiguranje
- DDOR (part of Unipol Group)
```

**4. 04_health_insurance.txt** (from character 1646)

```
2. Required documents:
   - Copy of passport (main page and page with residence permit sticker) or printout of biometric residence card (boravak)
   - Paid receipt for health card production (approximately 400 RSD per person, including children)
   - Completed application form (Form MA)
   - For sole proprietors: extract from the business register (Izvod o registraciji) and sometimes a tax office certificate
   - For employees: employer's confirmation of employment registration

3. Visit the appropriate RFZO branch and submit documents.
4. You will receive a temporary paper document that allows you to seek medical help immediately.
5. After processing, you receive a health insurance card (zdravstvena kartica).
```

---

## 14. [ru] сколько стоит сделать здравствену картицу и какое заявление там заполняют?

- **Answer should come from:** 04_health_insurance.txt
- **Expected fact:** Около 400 RSD на человека; форма MA
- **Retrieved:** 04_health_insurance.txt, 04_health_insurance.txt, 01_white_card_registration.txt, 04_health_insurance.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Стоимость изготовления здравствене картице составляет примерно 400 RSD на человека, включая детей. Заявление, которое заполняют, называется форма MA.

### The passages the prompt carried

**1. 04_health_insurance.txt** (from character 1646)

```
2. Required documents:
   - Copy of passport (main page and page with residence permit sticker) or printout of biometric residence card (boravak)
   - Paid receipt for health card production (approximately 400 RSD per person, including children)
   - Completed application form (Form MA)
   - For sole proprietors: extract from the business register (Izvod o registraciji) and sometimes a tax office certificate
   - For employees: employer's confirmation of employment registration

3. Visit the appropriate RFZO branch and submit documents.
4. You will receive a temporary paper document that allows you to seek medical help immediately.
5. After processing, you receive a health insurance card (zdravstvena kartica).
```

**2. 04_health_insurance.txt** (from character 2367)

```
HEALTH INSURANCE CARD (ZDRAVSTVENA KARTICA)
- The card proves your insurance status when visiting public healthcare facilities.
- It must be renewed periodically — check online if automatic renewal has occurred.
- If the card is not renewed automatically, there may be a tax debt. Pay off the debt, obtain a certificate of no debt from the tax office, and take it to RFZO to renew.

IMPORTANT BENEFIT: If a public clinic offers a doctor's appointment with a waiting period of more than 30 days, you can request written confirmation. With this document, you can see a private specialist and submit the receipt to RFZO for reimbursement.

EMERGENCY MEDICAL CARE FOR FOREIGNERS
All foreign citizens in Serbia have the right to emergency medical assistance regardless of insurance status. However, payment depends on bilateral agreements:
```

**3. 01_white_card_registration.txt** (from character 0)

```
WHITE CARD (BELI KARTON) — REGISTRATION OF FOREIGNER'S ADDRESS IN SERBIA

WHAT IS THE WHITE CARD?
The White Card (Beli karton) is the mandatory registration of a foreigner's residential address in Serbia. Every foreigner staying in Serbia must register their address with the police within 24 hours of arrival. This is the first and most important step for any foreigner entering Serbia. Without it, you cannot proceed with any other administrative procedures such as obtaining a temporary residence permit, opening a bank account, or registering for health insurance.

WHO MUST REGISTER?
All foreign nationals staying in the Republic of Serbia, regardless of the purpose or duration of their stay. This applies to tourists, business visitors, and long-term residents alike.
```

**4. 04_health_insurance.txt** (from character 925)

```
Public health insurance covers:
- Doctor visits at public health centers (dom zdravlja)
- Hospital treatment and surgeries
- Specialist consultations (with referral from a general practitioner)
- Prescription medications (some are free at pharmacies)
- Laboratory tests and diagnostics
- Maternity care

HOW TO REGISTER FOR PUBLIC HEALTH INSURANCE (OMS)
The application process involves several steps:

1. Determine the submission location based on your status:
   - Employees: submit at the RFZO branch at the location of your employer's headquarters
   - Sole proprietors: submit at the branch corresponding to your business's registered address
   - Unemployed and other categories: submit at your place of residence
```

---

## 15. [sr] dolazi mi rodjaka iz nemacke - da li joj ovde vazi njena zdravstvena kartica?

- **Answer should come from:** 04_health_insurance.txt
- **Expected fact:** Nemačka je među zemljama sa bilateralnim sporazumom; koristi se evropska kartica zdravstvenog osiguranja
- **Retrieved:** 04_health_insurance.txt, 04_health_insurance.txt, 01_white_card_registration.txt, 04_health_insurance.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [] grounded  [x] ungrounded  [ ] declined

**Notes:** Germany, the bilateral agreement and the EHIC all come from passage 4. The answer blurs which of that passage's two lists Germany belongs to by offering "specific forms" as an alternative, but it introduces nothing the passage does not print.

### The answer

Da, njena zdravstvena kartica iz Nemačke može važiti u Srbiji, jer postoji bilateralni zdravstveni sporazum između Nemačke i Srbije. Međutim, može biti potrebno da koristi Evropsku zdravstvenu karticu (EHIC) ili specifične obrasce za hitnu medicinsku pomoć.

### The passages the prompt carried

**1. 04_health_insurance.txt** (from character 1646)

```
2. Required documents:
   - Copy of passport (main page and page with residence permit sticker) or printout of biometric residence card (boravak)
   - Paid receipt for health card production (approximately 400 RSD per person, including children)
   - Completed application form (Form MA)
   - For sole proprietors: extract from the business register (Izvod o registraciji) and sometimes a tax office certificate
   - For employees: employer's confirmation of employment registration

3. Visit the appropriate RFZO branch and submit documents.
4. You will receive a temporary paper document that allows you to seek medical help immediately.
5. After processing, you receive a health insurance card (zdravstvena kartica).
```

**2. 04_health_insurance.txt** (from character 2367)

```
HEALTH INSURANCE CARD (ZDRAVSTVENA KARTICA)
- The card proves your insurance status when visiting public healthcare facilities.
- It must be renewed periodically — check online if automatic renewal has occurred.
- If the card is not renewed automatically, there may be a tax debt. Pay off the debt, obtain a certificate of no debt from the tax office, and take it to RFZO to renew.

IMPORTANT BENEFIT: If a public clinic offers a doctor's appointment with a waiting period of more than 30 days, you can request written confirmation. With this document, you can see a private specialist and submit the receipt to RFZO for reimbursement.

EMERGENCY MEDICAL CARE FOR FOREIGNERS
All foreign citizens in Serbia have the right to emergency medical assistance regardless of insurance status. However, payment depends on bilateral agreements:
```

**3. 01_white_card_registration.txt** (from character 1333)

```
HOW TO REGISTER — STEP BY STEP
1. The landlord (property owner) or accommodation provider must go to the nearest police station (policijska stanica) in the territory where the accommodation is located.
2. Required documents:
   - Foreigner's passport (original for inspection, photocopy to submit)
   - Landlord's personal ID card (lična karta)
   - Proof of property ownership or rental agreement
   - Landlord's written statement (izjava stanodavca) consenting to register the foreigner at their address
3. The police station issues the White Card (Beli karton) — a registration slip confirming the foreigner's registered address.
4. Keep this document safe — you will need it for virtually every subsequent administrative procedure.
```

**4. 04_health_insurance.txt** (from character 3004)

```
EMERGENCY MEDICAL CARE FOR FOREIGNERS
All foreign citizens in Serbia have the right to emergency medical assistance regardless of insurance status. However, payment depends on bilateral agreements:

Countries with bilateral health agreements (using European Health Insurance Card — EHIC):
- Austria, Bulgaria, Croatia, Hungary, Germany, Luxembourg, Slovakia, Slovenia, Czech Republic

Countries using specific bilateral forms:
- Belgium (Form BE/SRB 111), Netherlands (N/Y-111), Italy (IT-7), France (SE-21-03A/SE-21-04A), Montenegro (MNE/SRB-111), Bosnia and Herzegovina (BIH/SRB-111), North Macedonia (RM/SRB 111), Romania (Y/R 11), Turkey (TR/SRB 111)

Countries using insurance certificates:
- Poland and Great Britain — based on insurance certificates issued in their home country
```

---

## 16. [en] applying for boravak and they want proof of insurance. what actually counts as proof?

- **Answer should come from:** 04_health_insurance.txt
- **Expected fact:** RFZO confirmation; a private policy valid in Serbia; or international cover valid in Serbia
- **Retrieved:** 04_health_insurance.txt, 02_temporary_residence_permit.txt, 02_temporary_residence_permit.txt, 04_health_insurance.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [ ] grounded  [x] ungrounded  [ ] declined

**Notes:** The passages state that proof of health insurance valid in Serbia is required but never say what counts as proof; the enumeration (a policy, a letter from the insurer) is the agent's own.

### The answer

Proof of health insurance valid in Serbia can include a health insurance policy or a document from your insurance provider confirming your coverage.

### The passages the prompt carried

**1. 04_health_insurance.txt** (from character 1646)

```
2. Required documents:
   - Copy of passport (main page and page with residence permit sticker) or printout of biometric residence card (boravak)
   - Paid receipt for health card production (approximately 400 RSD per person, including children)
   - Completed application form (Form MA)
   - For sole proprietors: extract from the business register (Izvod o registraciji) and sometimes a tax office certificate
   - For employees: employer's confirmation of employment registration

3. Visit the appropriate RFZO branch and submit documents.
4. You will receive a temporary paper document that allows you to seek medical help immediately.
5. After processing, you receive a health insurance card (zdravstvena kartica).
```

**2. 02_temporary_residence_permit.txt** (from character 2675)

```
Note: Proof of means of subsistence and health insurance are NOT required for foreigners who are immediate family members of a Serbian citizen or who apply on the basis of employment.

APPLYING ONLINE VS IN PERSON
Online (recommended):
- Register at welcometoserbia.gov.rs
- Fill out the online application form
- Upload required documents
- Pay fees electronically
- Track application status through the portal
- Receive notifications in your eDatabox (eSanduče)

In person:
- Schedule an appointment at the competent police department by email
- Submit the application with all required documents
- The address book of regional police departments can be found on the MUP website (mup.gov.rs)
```

**3. 02_temporary_residence_permit.txt** (from character 2056)

```
REQUIRED DOCUMENTS (GENERAL)
- Valid passport (validity must exceed the requested residence period by at least 3 months)
- Proof of registered residential address in Serbia (White Card)
- Landlord's statement consenting to address registration
- Proof of sufficient means of subsistence (employment contract, bank statement, or other evidence)
- Proof of health insurance valid in Serbia
- Additional documents depending on the specific ground for residence (employment contract, enrollment certificate, property deed, etc.)
- Passport-size color photo (less than 6 months old, light background, face clearly visible)

Note: Proof of means of subsistence and health insurance are NOT required for foreigners who are immediate family members of a Serbian citizen or who apply on the basis of employment.
```

**4. 04_health_insurance.txt** (from character 2367)

```
HEALTH INSURANCE CARD (ZDRAVSTVENA KARTICA)
- The card proves your insurance status when visiting public healthcare facilities.
- It must be renewed periodically — check online if automatic renewal has occurred.
- If the card is not renewed automatically, there may be a tax debt. Pay off the debt, obtain a certificate of no debt from the tax office, and take it to RFZO to renew.

IMPORTANT BENEFIT: If a public clinic offers a doctor's appointment with a waiting period of more than 30 days, you can request written confirmation. With this document, you can see a private specialist and submit the receipt to RFZO for reimbursement.

EMERGENCY MEDICAL CARE FOR FOREIGNERS
All foreign citizens in Serbia have the right to emergency medical assistance regardless of insurance status. However, payment depends on bilateral agreements:
```

---

## 17. [en] im freelancing here. when do i actually have to file? is it every three months?

- **Answer should come from:** 05_tax_registration_pib.txt
- **Expected fact:** 1-30 April; 1-30 July; 1-30 October; 1-30 January
- **Retrieved:** 05_tax_registration_pib.txt, 05_tax_registration_pib.txt, 05_tax_registration_pib.txt, 05_tax_registration_pib.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Yes, you need to submit quarterly tax returns (PP OPO-K form) as a freelancer. The filing periods are as follows:
- Q1: April 1-30
- Q2: July 1-30
- Q3: October 1-30
- Q4: January 1-30 (of the following year)

### The passages the prompt carried

**1. 05_tax_registration_pib.txt** (from character 2862)

```
Important: You do NOT need a work permit to register as a freelancer and report income under the self-taxation regime. You only need a foreigner's registration number (EBS) or a PIB.

How to report freelancer income:
1. Register on the Freelancers portal: frilenseri.purs.gov.rs
2. Submit quarterly tax returns (PP OPO-K form):
   - Q1: April 1-30
   - Q2: July 1-30
   - Q3: October 1-30
   - Q4: January 1-30 (of the following year)
3. The portal automatically generates a payment slip with QR code for easy payment via mobile banking.
4. Convert foreign currency income to RSD using the National Bank of Serbia middle exchange rate on the payment date.

Freelancer tax rates:
- Income tax: 20% on the net taxable amount (after deducting normative costs)
- Normative costs (automatic deduction): depends on income type (typically 20-50%)
- Pension and disability insurance: 24% (mandatory)
- Health insurance: 10.3% (mandatory, unless covered by a bilateral agreement)
```

**2. 05_tax_registration_pib.txt** (from character 2398)

```
Freelancers (self-taxation):
Since December 2022, Serbia has regulated freelancer taxation through amendments to the Law on Personal Income Tax.

Who qualifies as a freelancer for tax purposes:
- Serbian residents earning income from foreign payers (companies, entrepreneurs, or individuals abroad)
- Foreign residents (non-residents) earning income in Serbia from foreign payers
- Anyone earning income from a domestic natural person who does not withhold taxes

Important: You do NOT need a work permit to register as a freelancer and report income under the self-taxation regime. You only need a foreigner's registration number (EBS) or a PIB.
```

**3. 05_tax_registration_pib.txt** (from character 1693)

```
ePOREZI (eTAXES) PORTAL
The ePorezi portal provides electronic tax services:
- Submit electronically signed tax return forms
- Monitor the status of submitted returns
- View the balance of taxpayer accounts
- Grant or revoke authorization for electronic services
- Available daily from 6:00 AM to midnight
- A TEST ENVIRONMENT is available for learning and verification

INCOME TAXES FOR FOREIGNERS

Employed foreigners:
- If you are employed by a Serbian company, your employer calculates and pays income tax and social contributions on your behalf through the Individual Tax Return (PPP-PD form).
- Tax rates: 10% income tax on salary, plus mandatory social contributions (pension: 25%, health: 10.3%)

Freelancers (self-taxation):
Since December 2022, Serbia has regulated freelancer taxation through amendments to the Law on Personal Income Tax.
```

**4. 05_tax_registration_pib.txt** (from character 4809)

```
USEFUL LINKS
- Tax Administration: purs.gov.rs
- eTaxes portal: eporezi.purs.gov.rs
- Freelancer portal: frilenseri.purs.gov.rs
- Freelancer guide (PDF): frilenseri.purs.gov.rs/upload/media/2024/3/28/11215/Guide_for_submitting_tax_application_and_payment_of_taxes_via_web_portal.pdf

Source: Tax Administration of the Republic of Serbia (purs.gov.rs), Welcome to Serbia portal (welcometoserbia.gov.rs)
```

---

## 18. [ru] сколько в итоге фрилансер отдает налогов и взносов?

- **Answer should come from:** 05_tax_registration_pib.txt
- **Expected fact:** 20% налог на доход; 24% пенсионные; 10.3% медицинские
- **Retrieved:** 05_tax_registration_pib.txt, 05_tax_registration_pib.txt, 05_tax_registration_pib.txt, 05_tax_registration_pib.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [ ] grounded  [x] ungrounded  [ ] declined

**Notes:** The three rates are correct, but the total 54.3% is arithmetic the passages do not carry, and the rates apply to different bases - income tax is on the net amount after normative costs.

### The answer

Фрилансер в Сербии отдает 20% налога на доход (на чистую налогооблагаемую сумму) и обязательные взносы: 24% на пенсионное и инвалидное страхование и 10.3% на медицинское страхование. В итоге это составляет 54.3% от налогооблагаемого дохода (20% + 24% + 10.3%).

### The passages the prompt carried

**1. 05_tax_registration_pib.txt** (from character 2862)

```
Important: You do NOT need a work permit to register as a freelancer and report income under the self-taxation regime. You only need a foreigner's registration number (EBS) or a PIB.

How to report freelancer income:
1. Register on the Freelancers portal: frilenseri.purs.gov.rs
2. Submit quarterly tax returns (PP OPO-K form):
   - Q1: April 1-30
   - Q2: July 1-30
   - Q3: October 1-30
   - Q4: January 1-30 (of the following year)
3. The portal automatically generates a payment slip with QR code for easy payment via mobile banking.
4. Convert foreign currency income to RSD using the National Bank of Serbia middle exchange rate on the payment date.

Freelancer tax rates:
- Income tax: 20% on the net taxable amount (after deducting normative costs)
- Normative costs (automatic deduction): depends on income type (typically 20-50%)
- Pension and disability insurance: 24% (mandatory)
- Health insurance: 10.3% (mandatory, unless covered by a bilateral agreement)
```

**2. 05_tax_registration_pib.txt** (from character 2398)

```
Freelancers (self-taxation):
Since December 2022, Serbia has regulated freelancer taxation through amendments to the Law on Personal Income Tax.

Who qualifies as a freelancer for tax purposes:
- Serbian residents earning income from foreign payers (companies, entrepreneurs, or individuals abroad)
- Foreign residents (non-residents) earning income in Serbia from foreign payers
- Anyone earning income from a domestic natural person who does not withhold taxes

Important: You do NOT need a work permit to register as a freelancer and report income under the self-taxation regime. You only need a foreigner's registration number (EBS) or a PIB.
```

**3. 05_tax_registration_pib.txt** (from character 1693)

```
ePOREZI (eTAXES) PORTAL
The ePorezi portal provides electronic tax services:
- Submit electronically signed tax return forms
- Monitor the status of submitted returns
- View the balance of taxpayer accounts
- Grant or revoke authorization for electronic services
- Available daily from 6:00 AM to midnight
- A TEST ENVIRONMENT is available for learning and verification

INCOME TAXES FOR FOREIGNERS

Employed foreigners:
- If you are employed by a Serbian company, your employer calculates and pays income tax and social contributions on your behalf through the Individual Tax Return (PPP-PD form).
- Tax rates: 10% income tax on salary, plus mandatory social contributions (pension: 25%, health: 10.3%)

Freelancers (self-taxation):
Since December 2022, Serbia has regulated freelancer taxation through amendments to the Law on Personal Income Tax.
```

**4. 05_tax_registration_pib.txt** (from character 4809)

```
USEFUL LINKS
- Tax Administration: purs.gov.rs
- eTaxes portal: eporezi.purs.gov.rs
- Freelancer portal: frilenseri.purs.gov.rs
- Freelancer guide (PDF): frilenseri.purs.gov.rs/upload/media/2024/3/28/11215/Guide_for_submitting_tax_application_and_payment_of_taxes_via_web_portal.pdf

Source: Tax Administration of the Republic of Serbia (purs.gov.rs), Welcome to Serbia portal (welcometoserbia.gov.rs)
```

---

## 19. [sr] treba li mi radna dozvola ako hocu da radim kao frilenser?

- **Answer should come from:** 05_tax_registration_pib.txt
- **Expected fact:** Ne; dovoljan je EBS broj ili PIB
- **Retrieved:** 05_tax_registration_pib.txt, 04_health_insurance.txt, 02_temporary_residence_permit.txt, 05_tax_registration_pib.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Ne, ne treba vam radna dozvola da biste se registrovali kao frilenser i prijavili prihod pod režimom samoporezivanja. Potrebni su vam samo broj stranog registracije (EBS) ili PIB.

### The passages the prompt carried

**1. 05_tax_registration_pib.txt** (from character 2862)

```
Important: You do NOT need a work permit to register as a freelancer and report income under the self-taxation regime. You only need a foreigner's registration number (EBS) or a PIB.

How to report freelancer income:
1. Register on the Freelancers portal: frilenseri.purs.gov.rs
2. Submit quarterly tax returns (PP OPO-K form):
   - Q1: April 1-30
   - Q2: July 1-30
   - Q3: October 1-30
   - Q4: January 1-30 (of the following year)
3. The portal automatically generates a payment slip with QR code for easy payment via mobile banking.
4. Convert foreign currency income to RSD using the National Bank of Serbia middle exchange rate on the payment date.

Freelancer tax rates:
- Income tax: 20% on the net taxable amount (after deducting normative costs)
- Normative costs (automatic deduction): depends on income type (typically 20-50%)
- Pension and disability insurance: 24% (mandatory)
- Health insurance: 10.3% (mandatory, unless covered by a bilateral agreement)
```

**2. 04_health_insurance.txt** (from character 1646)

```
2. Required documents:
   - Copy of passport (main page and page with residence permit sticker) or printout of biometric residence card (boravak)
   - Paid receipt for health card production (approximately 400 RSD per person, including children)
   - Completed application form (Form MA)
   - For sole proprietors: extract from the business register (Izvod o registraciji) and sometimes a tax office certificate
   - For employees: employer's confirmation of employment registration

3. Visit the appropriate RFZO branch and submit documents.
4. You will receive a temporary paper document that allows you to seek medical help immediately.
5. After processing, you receive a health insurance card (zdravstvena kartica).
```

**3. 02_temporary_residence_permit.txt** (from character 4124)

```
SINGLE PERMIT (UNIFIED PERMIT FOR RESIDENCE AND WORK)
Since February 2024, foreigners employed in Serbia receive a Single Permit that combines both residence and work authorization. Key points:
- Applied for through the Welcome to Serbia portal
- Standard processing time: 19 days
- Visa-free nationals must apply for a Single Permit (Visa D is no longer available for most visa-free nationals who want to work)
- The employer can submit the application on behalf of the employee with a power of attorney
- Changes of employer or basis of employment can be submitted through the portal

CERTIFICATE OF INITIATED PROCEDURE
After submitting your application, you receive a certificate confirming that the procedure has been initiated. This certificate includes your foreigner registration number (EBS), which you need for exercising rights and obligations in Serbia (opening a bank account, registering for health insurance, etc.).
```

**4. 05_tax_registration_pib.txt** (from character 2398)

```
Freelancers (self-taxation):
Since December 2022, Serbia has regulated freelancer taxation through amendments to the Law on Personal Income Tax.

Who qualifies as a freelancer for tax purposes:
- Serbian residents earning income from foreign payers (companies, entrepreneurs, or individuals abroad)
- Foreign residents (non-residents) earning income in Serbia from foreign payers
- Anyone earning income from a domestic natural person who does not withhold taxes

Important: You do NOT need a work permit to register as a freelancer and report income under the self-taxation regime. You only need a foreigner's registration number (EBS) or a PIB.
```

---

## 20. [en] im doing visa runs every 90 days and working remotely. do i owe tax here?

- **Answer should come from:** 05_tax_registration_pib.txt
- **Expected fact:** Yes - earning income in Serbia can create tax obligations regardless of visa status
- **Retrieved:** 05_tax_registration_pib.txt, 05_tax_registration_pib.txt, 05_tax_registration_pib.txt, 05_tax_registration_pib.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Yes, earning income while in Serbia may create tax obligations regardless of your visa status. You should report your income and may need to pay taxes.

### The passages the prompt carried

**1. 05_tax_registration_pib.txt** (from character 3834)

```
If you paid tax abroad on the same income, you can claim a tax credit (up to the amount of Serbian tax liability) to avoid double taxation. Serbia has double taxation avoidance agreements with over 65 countries.

VISA RUN AND TAX IMPLICATIONS
Foreigners who stay in Serbia on visa-free entry (up to 90 days in 180 days) and perform "visa runs" (exiting and re-entering the country to reset the stay period) should be aware that earning income while in Serbia may create tax obligations regardless of visa status.

IMPORTANT NOTES
- The PIB is used as the VAT number for businesses engaged in taxable transactions.
- Verify your PIB through the Tax Administration's official verification tool.
- If you are a sole proprietor, you must register for personal salary payments within 5 days of receiving your PIB.
- All tax obligations can be managed electronically through the ePorezi portal.
- Late tax payments incur interest calculated per the National Bank of Serbia rates.
```

**2. 05_tax_registration_pib.txt** (from character 1693)

```
ePOREZI (eTAXES) PORTAL
The ePorezi portal provides electronic tax services:
- Submit electronically signed tax return forms
- Monitor the status of submitted returns
- View the balance of taxpayer accounts
- Grant or revoke authorization for electronic services
- Available daily from 6:00 AM to midnight
- A TEST ENVIRONMENT is available for learning and verification

INCOME TAXES FOR FOREIGNERS

Employed foreigners:
- If you are employed by a Serbian company, your employer calculates and pays income tax and social contributions on your behalf through the Individual Tax Return (PPP-PD form).
- Tax rates: 10% income tax on salary, plus mandatory social contributions (pension: 25%, health: 10.3%)

Freelancers (self-taxation):
Since December 2022, Serbia has regulated freelancer taxation through amendments to the Law on Personal Income Tax.
```

**3. 05_tax_registration_pib.txt** (from character 2862)

```
Important: You do NOT need a work permit to register as a freelancer and report income under the self-taxation regime. You only need a foreigner's registration number (EBS) or a PIB.

How to report freelancer income:
1. Register on the Freelancers portal: frilenseri.purs.gov.rs
2. Submit quarterly tax returns (PP OPO-K form):
   - Q1: April 1-30
   - Q2: July 1-30
   - Q3: October 1-30
   - Q4: January 1-30 (of the following year)
3. The portal automatically generates a payment slip with QR code for easy payment via mobile banking.
4. Convert foreign currency income to RSD using the National Bank of Serbia middle exchange rate on the payment date.

Freelancer tax rates:
- Income tax: 20% on the net taxable amount (after deducting normative costs)
- Normative costs (automatic deduction): depends on income type (typically 20-50%)
- Pension and disability insurance: 24% (mandatory)
- Health insurance: 10.3% (mandatory, unless covered by a bilateral agreement)
```

**4. 05_tax_registration_pib.txt** (from character 2398)

```
Freelancers (self-taxation):
Since December 2022, Serbia has regulated freelancer taxation through amendments to the Law on Personal Income Tax.

Who qualifies as a freelancer for tax purposes:
- Serbian residents earning income from foreign payers (companies, entrepreneurs, or individuals abroad)
- Foreign residents (non-residents) earning income in Serbia from foreign payers
- Anyone earning income from a domestic natural person who does not withhold taxes

Important: You do NOT need a work permit to register as a freelancer and report income under the self-taxation regime. You only need a foreigner's registration number (EBS) or a PIB.
```

---

## 21. [en] why is there heating on my infostan bill in july?

- **Answer should come from:** 06_utility_bills_payments.txt
- **Expected fact:** Heating is charged year-round at about 119 RSD per m² per month
- **Retrieved:** 06_utility_bills_payments.txt, 06_utility_bills_payments.txt, 06_utility_bills_payments.txt, 06_utility_bills_payments.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Heating is charged year-round on your Infostan bill, not just during the heating season.

### The passages the prompt carried

**1. 06_utility_bills_payments.txt** (from character 1571)

```
What the Infostan bill includes:
- Heating (JKP "Beogradske elektrane"): charged at approximately 119 RSD per square meter per month. Heating is charged year-round (not only during the heating season).
- Water supply and sewage (JKP "Beogradski vodovod"): based on consumption
- Garbage collection (JKP "Gradska čistoća"): approximately 300 RSD per month
- Building maintenance (JP "Gradsko Stambeno"): 17.12 RSD/m² with elevator, 10.72 RSD/m² without elevator
- Building cleaning: approximately 218 RSD per apartment
- Common electricity for the building (EPS Distribucija)
- Bio-energy and other small ecological fees

Discount: Pay before the 15th of the month to receive approximately 5-10% discount on the next bill.
```

**2. 06_utility_bills_payments.txt** (from character 907)

```
How to manage your EPS account:
- Register on the EPS portal "Uvid u račun" (Bill overview) at eps.rs to receive e-bills
- Download the EPS mobile app (iOS and Android) to view consumption, pay bills, and check history
- Switching to electronic bills gives a discount of 50 RSD on the next bill
- You can view payment history for the past 3 years online
- Prepayment option: pay in advance for the next month to receive a small discount

INFOSTAN (BELGRADE) — COMMUNAL SERVICES
JKP "Infostan tehnologije" is the unified billing company for communal services in Belgrade, operating since 1977. It collects payments for multiple utility providers on a single bill.
```

**3. 06_utility_bills_payments.txt** (from character 5401)

```
TIPS FOR FOREIGNERS
- Always keep the reference number (poziv na broj) from your bills — it identifies your account.
- Set up e-bills where possible to avoid missing paper bills that may arrive late.
- If you move apartments, notify utility companies to transfer or close accounts.
- Utility contracts are usually in the landlord's name. Clarify with your landlord how bills are handled.
- Check with your landlord whether utilities are included in rent or paid separately.
- In winter, electricity costs rise significantly if you have electric heating.

USEFUL LINKS
- EPS (electricity): eps.rs
- Infostan (Belgrade utilities): infostan.rs
- Informatika (Novi Sad utilities): nsinfo.co.rs
- Serbian Post Office (bill payments): posta.rs
- eSanduče: esanduce.rs

Source: EPS (eps.rs), Infostan (infostan.rs), City Expert blog (cityexpert.rs), Serbian Post (posta.rs)
```

**4. 06_utility_bills_payments.txt** (from character 0)

```
UTILITY BILLS AND PAYMENTS IN SERBIA — GUIDE FOR FOREIGNERS

OVERVIEW
Living in Serbia means dealing with several regular utility bills. The main ones are electricity (EPS), communal services including heating and water (Infostan in Belgrade, Informatika in Novi Sad), internet/phone, and building maintenance. This guide explains what each bill covers, how to read them, and how to pay.

ELECTRICITY — EPS (ELEKTROPRIVREDA SRBIJE)
EPS is the sole electricity provider in Serbia. Every household has an EPS account.

What the bill includes:
- Electricity consumption measured in kWh
- Charges may vary by tariff zone (day/night rates for dual-tariff meters)
- Monthly fixed connection fee

Typical costs:
- Summer months (no electric heating): 3,000-3,500 RSD (25-30 EUR)
- Winter months (with electric heating): 15,000-18,000 RSD (130-155 EUR)
- Serbia has among the cheapest electricity prices in Europe
```

---

## 22. [ru] оплатил счет а деньги не зачли. говорят из-за позива на брой - что это такое?

- **Answer should come from:** 06_utility_bills_payments.txt
- **Expected fact:** Уникальный номер плательщика; без него платёж не зачислят на нужный счёт
- **Retrieved:** 06_utility_bills_payments.txt, 06_utility_bills_payments.txt, 03_opening_bank_account.txt, 06_utility_bills_payments.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Позив на број (Reference number) — это ваш уникальный номер клиента/счета, который необходимо указать при оплате, чтобы платеж был правильно зачислен на ваш счет. Если он указан неверно, деньги могут не быть зачислены. Убедитесь, что вы правильно ввели этот номер при оплате.

### The passages the prompt carried

**1. 06_utility_bills_payments.txt** (from character 3856)

```
Method 3: Bank — Standing Order
- Set up automatic monthly payments (trajni nalog) through your bank
- Available at Erste Bank, OTP Bank, Vojvođanska Banka, NLB Banka, Poštanska Štedionica, and others
- Banca Intesa offers commission-free bill payments

Method 4: Online / e-Banking
- Pay through your bank's online banking platform
- Use the payment slip details from the bill (recipient account number, reference number, amount)
- IPS QR code payments available for instant settlement

Method 5: eSanduče (Electronic Mailbox)
- A government electronic service for receiving and paying bills online
- Consolidates Infostan and MTS bills
- Pay via Visa, MasterCard, Dina, Maestro, or American Express cards
- Access through mobile app or web portal

Method 6: EPS Portal / App
- Pay electricity bills directly on the EPS website or mobile app
- Supports IPS Scan payments and card payments (Dina, Visa, MasterCard)
```

**2. 06_utility_bills_payments.txt** (from character 4606)

```
Method 6: EPS Portal / App
- Pay electricity bills directly on the EPS website or mobile app
- Supports IPS Scan payments and card payments (Dina, Visa, MasterCard)

UNDERSTANDING A SERBIAN UTILITY BILL (NALOG ZA UPLATU)
A typical Serbian payment slip contains:
- Platilac (Payer): your name and address
- Svrha plaćanja (Payment purpose): description of what you are paying for
- Primalac (Recipient): the utility company name and address
- Šifra plaćanja (Payment code): a 3-digit code (e.g., 289 for utility payments)
- Valuta (Currency): RSD
- Iznos (Amount): the amount to pay
- Račun primaoca (Recipient account): the bank account number to pay to
- Poziv na broj (Reference number): your unique customer/account number — this is critical for the payment to be attributed to your account
```

**3. 03_opening_bank_account.txt** (from character 4693)

```
BILL PAYMENTS THROUGH BANK ACCOUNTS
Once you have a bank account, you can set up:
- Standing orders for recurring utility payments (EPS electricity, Infostan, internet)
- e-Banking payments using payment slips (nalog za uplatu)
- IPS QR code payments for instant settlement

LEGAL BASIS
Law on Foreign Exchange Operations ("Official Gazette of RS" Nos. 62/2006, 31/2011, 119/2012, 139/2014, and 30/2018)
Law on Payment Services ("Official Gazette of RS" Nos. 139/2014 and 44/2018)
Law on Prevention of Money Laundering and Terrorism Financing

USEFUL LINKS
- National Bank of Serbia: nbs.rs
- Welcome to Serbia — Banking info: welcometoserbia.gov.rs/opening-bank-accounts-for-residents-and-non-residents

Source: Welcome to Serbia portal (welcometoserbia.gov.rs), National Bank of Serbia (nbs.rs), Wise.com, Zunic Law
```

**4. 06_utility_bills_payments.txt** (from character 2192)

```
Discount: Pay before the 15th of the month to receive approximately 5-10% discount on the next bill.

INFORMATIKA (NOVI SAD) — COMMUNAL SERVICES
In Novi Sad, JKP "Informatika" performs a similar unified billing function.
- Office: 3 Car Lazar Blvd., 21102 Novi Sad
- Contact Centre: 0800 222 021
- Late payments incur interest calculated per the National Bank of Serbia rate.
- Enforced collection may be initiated for unpaid bills after a warning letter.

INTERNET AND PHONE
Major providers:
- MTS (Telekom Srbija): largest provider, offers fiber optic, ADSL, mobile, and TV packages
- Yettel (formerly Telenor): mobile and internet services
- A1 (formerly Vip Mobile): mobile and home internet
- SBB (Serbia Broadband): cable internet and TV

SIM card: You can purchase a SIM card without documents, but activation and full use require a valid passport. Visit the operator's office to link the card to your passport.
```

---

## 23. [sr] isplati li se platiti infostan ranije u mesecu?

- **Answer should come from:** 06_utility_bills_payments.txt
- **Expected fact:** Popust oko 5-10% ako se plati pre 15. u mesecu
- **Retrieved:** 06_utility_bills_payments.txt, 06_utility_bills_payments.txt, 06_utility_bills_payments.txt, 06_utility_bills_payments.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Da, isplati se platiti Infostan ranije u mesecu jer možete dobiti približno 5-10% popusta na sledeći račun ako platite pre 15. u mesecu.

### The passages the prompt carried

**1. 06_utility_bills_payments.txt** (from character 2192)

```
Discount: Pay before the 15th of the month to receive approximately 5-10% discount on the next bill.

INFORMATIKA (NOVI SAD) — COMMUNAL SERVICES
In Novi Sad, JKP "Informatika" performs a similar unified billing function.
- Office: 3 Car Lazar Blvd., 21102 Novi Sad
- Contact Centre: 0800 222 021
- Late payments incur interest calculated per the National Bank of Serbia rate.
- Enforced collection may be initiated for unpaid bills after a warning letter.

INTERNET AND PHONE
Major providers:
- MTS (Telekom Srbija): largest provider, offers fiber optic, ADSL, mobile, and TV packages
- Yettel (formerly Telenor): mobile and internet services
- A1 (formerly Vip Mobile): mobile and home internet
- SBB (Serbia Broadband): cable internet and TV

SIM card: You can purchase a SIM card without documents, but activation and full use require a valid passport. Visit the operator's office to link the card to your passport.
```

**2. 06_utility_bills_payments.txt** (from character 907)

```
How to manage your EPS account:
- Register on the EPS portal "Uvid u račun" (Bill overview) at eps.rs to receive e-bills
- Download the EPS mobile app (iOS and Android) to view consumption, pay bills, and check history
- Switching to electronic bills gives a discount of 50 RSD on the next bill
- You can view payment history for the past 3 years online
- Prepayment option: pay in advance for the next month to receive a small discount

INFOSTAN (BELGRADE) — COMMUNAL SERVICES
JKP "Infostan tehnologije" is the unified billing company for communal services in Belgrade, operating since 1977. It collects payments for multiple utility providers on a single bill.
```

**3. 06_utility_bills_payments.txt** (from character 1571)

```
What the Infostan bill includes:
- Heating (JKP "Beogradske elektrane"): charged at approximately 119 RSD per square meter per month. Heating is charged year-round (not only during the heating season).
- Water supply and sewage (JKP "Beogradski vodovod"): based on consumption
- Garbage collection (JKP "Gradska čistoća"): approximately 300 RSD per month
- Building maintenance (JP "Gradsko Stambeno"): 17.12 RSD/m² with elevator, 10.72 RSD/m² without elevator
- Building cleaning: approximately 218 RSD per apartment
- Common electricity for the building (EPS Distribucija)
- Bio-energy and other small ecological fees

Discount: Pay before the 15th of the month to receive approximately 5-10% discount on the next bill.
```

**4. 06_utility_bills_payments.txt** (from character 3856)

```
Method 3: Bank — Standing Order
- Set up automatic monthly payments (trajni nalog) through your bank
- Available at Erste Bank, OTP Bank, Vojvođanska Banka, NLB Banka, Poštanska Štedionica, and others
- Banca Intesa offers commission-free bill payments

Method 4: Online / e-Banking
- Pay through your bank's online banking platform
- Use the payment slip details from the bill (recipient account number, reference number, amount)
- IPS QR code payments available for instant settlement

Method 5: eSanduče (Electronic Mailbox)
- A government electronic service for receiving and paying bills online
- Consolidates Infostan and MTS bills
- Pay via Visa, MasterCard, Dina, Maestro, or American Express cards
- Access through mobile app or web portal

Method 6: EPS Portal / App
- Pay electricity bills directly on the EPS website or mobile app
- Supports IPS Scan payments and card payments (Dina, Visa, MasterCard)
```

---

## 24. [ru] где дешевле всего заплатить за коммуналку наличными?

- **Answer should come from:** 06_utility_bills_payments.txt
- **Expected fact:** На почте; комиссия от 65 RSD за платёж
- **Retrieved:** 06_utility_bills_payments.txt, 06_utility_bills_payments.txt, 06_utility_bills_payments.txt, 06_utility_bills_payments.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Самый дешевый способ оплаты коммунальных услуг наличными — это почта (Pošta), где комиссия начинается от 65 RSD за платеж.

### The passages the prompt carried

**1. 06_utility_bills_payments.txt** (from character 2937)

```
SIM card: You can purchase a SIM card without documents, but activation and full use require a valid passport. Visit the operator's office to link the card to your passport.

BUILDING MAINTENANCE (ODRŽAVANJE ZGRADE)
- A separate fee for common areas, elevator maintenance, and building cleaning
- Charged per square meter of your apartment
- Usually collected through the Infostan/Informatika bill or separately by the building management (skupština stanara)

HOW TO PAY UTILITY BILLS

Method 1: Post Office (Pošta)
- Bring the paper bill (nalog za uplatu) to any post office
- Pay in cash (RSD only)
- Lowest commission fees (starting from 65 RSD per payment)
- Available at all post offices across Serbia

Method 2: Exchange Offices (Menjačnica)
- Many exchange offices (such as Exclusive Change) accept bill payments
- Commission: typically 0.60%, minimum 65 RSD
- Available evenings and weekends at many locations
```

**2. 06_utility_bills_payments.txt** (from character 2192)

```
Discount: Pay before the 15th of the month to receive approximately 5-10% discount on the next bill.

INFORMATIKA (NOVI SAD) — COMMUNAL SERVICES
In Novi Sad, JKP "Informatika" performs a similar unified billing function.
- Office: 3 Car Lazar Blvd., 21102 Novi Sad
- Contact Centre: 0800 222 021
- Late payments incur interest calculated per the National Bank of Serbia rate.
- Enforced collection may be initiated for unpaid bills after a warning letter.

INTERNET AND PHONE
Major providers:
- MTS (Telekom Srbija): largest provider, offers fiber optic, ADSL, mobile, and TV packages
- Yettel (formerly Telenor): mobile and internet services
- A1 (formerly Vip Mobile): mobile and home internet
- SBB (Serbia Broadband): cable internet and TV

SIM card: You can purchase a SIM card without documents, but activation and full use require a valid passport. Visit the operator's office to link the card to your passport.
```

**3. 06_utility_bills_payments.txt** (from character 3856)

```
Method 3: Bank — Standing Order
- Set up automatic monthly payments (trajni nalog) through your bank
- Available at Erste Bank, OTP Bank, Vojvođanska Banka, NLB Banka, Poštanska Štedionica, and others
- Banca Intesa offers commission-free bill payments

Method 4: Online / e-Banking
- Pay through your bank's online banking platform
- Use the payment slip details from the bill (recipient account number, reference number, amount)
- IPS QR code payments available for instant settlement

Method 5: eSanduče (Electronic Mailbox)
- A government electronic service for receiving and paying bills online
- Consolidates Infostan and MTS bills
- Pay via Visa, MasterCard, Dina, Maestro, or American Express cards
- Access through mobile app or web portal

Method 6: EPS Portal / App
- Pay electricity bills directly on the EPS website or mobile app
- Supports IPS Scan payments and card payments (Dina, Visa, MasterCard)
```

**4. 06_utility_bills_payments.txt** (from character 907)

```
How to manage your EPS account:
- Register on the EPS portal "Uvid u račun" (Bill overview) at eps.rs to receive e-bills
- Download the EPS mobile app (iOS and Android) to view consumption, pay bills, and check history
- Switching to electronic bills gives a discount of 50 RSD on the next bill
- You can view payment history for the past 3 years online
- Prepayment option: pay in advance for the next month to receive a small discount

INFOSTAN (BELGRADE) — COMMUNAL SERVICES
JKP "Infostan tehnologije" is the unified billing company for communal services in Belgrade, operating since 1977. It collects payments for multiple utility providers on a single bill.
```

---

## 25. [en] can i stay the full 90 days and then just come back a week later?

- **Answer should come from:** 07_visa_regime_entry.txt
- **Expected fact:** No - the 90 days are cumulative within any rolling 180-day period
- **Retrieved:** 07_visa_regime_entry.txt, 07_visa_regime_entry.txt, 02_temporary_residence_permit.txt, 02_temporary_residence_permit.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

No, you cannot re-enter Serbia until 90 more days have passed after staying the full 90 days.

### The passages the prompt carried

**1. 07_visa_regime_entry.txt** (from character 938)

```
Important: The 90/180 rule means you can stay a maximum of 90 days in any rolling 180-day period. This is not 90 consecutive days — it is cumulative. If you stay 90 days and leave, you cannot re-enter until 90 more days have passed.

VISA RUN
A "visa run" is the common practice of leaving Serbia before the 90-day limit and re-entering to reset the stay period. Common visa run destinations from Serbia include:
- Bosnia and Herzegovina (closest border crossings)
- North Macedonia
- Hungary
- Romania
- Bulgaria

Procedure: Drive or take transport to the border, exit Serbia (get an exit stamp), enter the neighboring country briefly, then re-enter Serbia (get a new entry stamp). Some organized visa runs involve walking across a border checkpoint and returning the same day.
```

**2. 07_visa_regime_entry.txt** (from character 1718)

```
Warning: Serbian border authorities are aware of this practice. While it is technically legal if done within the 90/180 rule, repeatedly doing so without legitimate purpose may raise questions. If you plan to stay long-term, applying for a temporary residence permit is strongly recommended.

VISA TYPES

Type C — Short-Stay Visa:
- For stays up to 90 days
- For tourism, business visits, family visits, medical treatment, etc.
- Required for nationals of countries that do not have visa-free agreements with Serbia
- Applied for at Serbian embassies/consulates abroad or online via welcometoserbia.gov.rs
```

**3. 02_temporary_residence_permit.txt** (from character 5055)

```
IMPORTANT NOTES
- You must stay in Serbia in accordance with the purpose for which residence was granted.
- If you entered with a Visa D, submit the residence application before the visa expires.
- Minor foreigners (under 18) must apply in person with a parent or guardian.
- All notifications regarding your application are delivered through the eDatabox if you applied online.
- The biometric residence permit card also serves as an identity document in Serbia.

AFTER 3 YEARS — PERMANENT RESIDENCE
After 3 continuous years of temporary residence, you may be eligible to apply for permanent residence (stalni boravak), which is granted indefinitely and issued as a foreigner ID card valid for 5 years (renewable).

LEGAL BASIS
Law on Foreigners ("Official Gazette of RS" No. 24/2018, 31/2019, and 62/2023)
Law on Employment of Foreigners
```

**4. 02_temporary_residence_permit.txt** (from character 0)

```
TEMPORARY RESIDENCE PERMIT (PRIVREMENI BORAVAK) IN SERBIA

WHAT IS A TEMPORARY RESIDENCE PERMIT?
A temporary residence permit is an official authorization that allows a foreign citizen to legally reside in the Republic of Serbia for a period longer than 90 days. Since February 2024, Serbia introduced the Single Permit (Jedinstvena dozvola), which combines both residence and work authorization in one document. The permit is issued as a biometric ID card.

WHO NEEDS A TEMPORARY RESIDENCE PERMIT?
Any foreign national who plans to stay in Serbia for more than 90 days within a 180-day period must apply for a temporary residence permit. Citizens of visa-free countries can stay up to 90 days without a permit, but must apply before this period expires if they wish to remain longer.
```

---

## 26. [ru] что за DTA такое и с какого времени оно вообще работает?

- **Answer should come from:** 07_visa_regime_entry.txt
- **Expected fact:** Электронная виза с QR-кодом; введена 23 апреля 2025
- **Retrieved:** 07_visa_regime_entry.txt, 08_egovernment_digital_services.txt, 02_temporary_residence_permit.txt, 01_white_card_registration.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Digital Travel Authorization (DTA) — это электронная виза, выданная в цифровом формате для граждан eligible стран. Она была введена 23 апреля 2025 года.

### The passages the prompt carried

**1. 07_visa_regime_entry.txt** (from character 2325)

```
Type D — Long-Stay Visa:
- For stays between 90 and 180 days
- Required for nationals who need a visa to enter Serbia and intend to apply for temporary residence
- Grants entry and residence rights during the visa validity
- If issued on the basis of employment, it simultaneously serves as permission for entry, residence, and work for up to 180 days
- Since February 2024, Visa D is no longer available to most visa-free nationals who want to work in Serbia — they must apply for a Single Permit instead

Digital Travel Authorization (DTA):
- Introduced on April 23, 2025
- An electronic visa issued in digital form to nationals of eligible countries
- Applied for through the Welcome to Serbia portal
- Contains a QR code for border check verification
- Security features comply with ICAO standards for Machine Readable Travel Documents
```

**2. 08_egovernment_digital_services.txt** (from character 2555)

```
eFOREIGNER PORTAL
Website: eforeigner.welcometoserbia.gov.rs

Specifically for temporary residence and Single Permit applications:
- Submit residence applications electronically
- Submit changes of employer or employment basis
- Track application progress
- Receive decisions through the eDatabox
- Access help and FAQs

Important notes:
- Applications for minors (under 18) cannot be submitted online — must be done in person
- If the system asks for a Visa ID during extension and you do not have one, contact support at office@ite.gov.rs to have your account reset with your EBS number
- Once submitted, applications cannot be cancelled through the portal

ePOREZI PORTAL (eTAXES)
Website: eporezi.purs.gov.rs

The Tax Administration's electronic services portal:
- Submit tax returns electronically
- Monitor submitted returns
- View taxpayer account balances
- Grant/revoke authorization for tax representatives
- Available daily 6:00 AM to midnight
- Test environment available for practice
```

**3. 02_temporary_residence_permit.txt** (from character 4124)

```
SINGLE PERMIT (UNIFIED PERMIT FOR RESIDENCE AND WORK)
Since February 2024, foreigners employed in Serbia receive a Single Permit that combines both residence and work authorization. Key points:
- Applied for through the Welcome to Serbia portal
- Standard processing time: 19 days
- Visa-free nationals must apply for a Single Permit (Visa D is no longer available for most visa-free nationals who want to work)
- The employer can submit the application on behalf of the employee with a power of attorney
- Changes of employer or basis of employment can be submitted through the portal

CERTIFICATE OF INITIATED PROCEDURE
After submitting your application, you receive a certificate confirming that the procedure has been initiated. This certificate includes your foreigner registration number (EBS), which you need for exercising rights and obligations in Serbia (opening a bank account, registering for health insurance, etc.).
```

**4. 01_white_card_registration.txt** (from character 776)

```
WHO IS RESPONSIBLE FOR REGISTRATION?
- Hotels, hostels, boarding houses, and other accommodation facilities register the foreigner's stay electronically through the eTourist application.
- If you are staying in private accommodation (rented apartment, staying with friends or family), the landlord or the person providing accommodation is legally obligated to register the foreigner's stay with the competent authority within 24 hours.
- The foreigner's stay can also be registered electronically on the eAdministration (eUprava) portal at euprava.gov.rs.
```

---

## 27. [sr] gde ljudi idu na viza-ran iz beograda?

- **Answer should come from:** 07_visa_regime_entry.txt
- **Expected fact:** Bosna i Hercegovina; Severna Makedonija; Mađarska; Rumunija; Bugarska
- **Retrieved:** 07_visa_regime_entry.txt, 07_visa_regime_entry.txt, 07_visa_regime_entry.txt, 07_visa_regime_entry.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Ljudi obično idu na viza-ran iz Beograda u Bosnu i Hercegovinu, Severnu Makedoniju, Mađarsku, Rumuniju ili Bugarsku.

### The passages the prompt carried

**1. 07_visa_regime_entry.txt** (from character 938)

```
Important: The 90/180 rule means you can stay a maximum of 90 days in any rolling 180-day period. This is not 90 consecutive days — it is cumulative. If you stay 90 days and leave, you cannot re-enter until 90 more days have passed.

VISA RUN
A "visa run" is the common practice of leaving Serbia before the 90-day limit and re-entering to reset the stay period. Common visa run destinations from Serbia include:
- Bosnia and Herzegovina (closest border crossings)
- North Macedonia
- Hungary
- Romania
- Bulgaria

Procedure: Drive or take transport to the border, exit Serbia (get an exit stamp), enter the neighboring country briefly, then re-enter Serbia (get a new entry stamp). Some organized visa runs involve walking across a border checkpoint and returning the same day.
```

**2. 07_visa_regime_entry.txt** (from character 1718)

```
Warning: Serbian border authorities are aware of this practice. While it is technically legal if done within the 90/180 rule, repeatedly doing so without legitimate purpose may raise questions. If you plan to stay long-term, applying for a temporary residence permit is strongly recommended.

VISA TYPES

Type C — Short-Stay Visa:
- For stays up to 90 days
- For tourism, business visits, family visits, medical treatment, etc.
- Required for nationals of countries that do not have visa-free agreements with Serbia
- Applied for at Serbian embassies/consulates abroad or online via welcometoserbia.gov.rs
```

**3. 07_visa_regime_entry.txt** (from character 3166)

```
Required documents for visa application:
- Valid passport
- Passport-size photo (color, recent, light background)
- Proof of purpose of visit
- Proof of sufficient funds for accommodation and living expenses
- Proof of health insurance
- Evidence of intent to leave Serbia before visa expiry (return ticket, etc.)

For Visa D, additional documents matching the intended temporary residence basis are required (employment contract, enrollment certificate, property deed, etc.).

HOW TO APPLY FOR A VISA
1. Online: Through the Welcome to Serbia portal (evisa.welcometoserbia.gov.rs)
   - Register an account on the eID portal (eid.gov.rs)
   - Fill out the application form
   - Upload required documents
   - Pay the fee
   - Track status online

2. In person: At the Embassy or Consulate General of the Republic of Serbia in your country
   - Submit application with all required documents
   - Attend an interview if required
   - Processing time varies
```

**4. 07_visa_regime_entry.txt** (from character 0)

```
VISA REGIME AND ENTRY REQUIREMENTS FOR SERBIA

OVERVIEW
Serbia has a relatively open visa policy. Many nationalities can enter visa-free for short stays. For longer stays, various visa types and residence permits are available. The entry requirements and visa regime are administered by the Ministry of Foreign Affairs and the Ministry of Interior.

GENERAL ENTRY REQUIREMENTS
When entering Serbia, all visitors must:
- Hold a valid passport (or national ID card for citizens of certain countries)
- Have sufficient funds for the duration of stay
- Have a return ticket or proof of onward travel (may be requested)
- Not be on the entry ban list
- Have no security concerns flagged by Serbian authorities

VISA-FREE ENTRY
Most foreigners, including citizens of the USA, UK, EU/EEA countries, Canada, Australia, and many others, can enter Serbia visa-free for up to 90 days within a 180-day period (counting from the first day of entry).
```

---

## 28. [en] is the welcome to serbia site in english or serbian only?

- **Answer should come from:** 08_egovernment_digital_services.txt
- **Expected fact:** Serbian in Cyrillic and Latin; English; Russian; Chinese
- **Retrieved:** 08_egovernment_digital_services.txt, 07_visa_regime_entry.txt, 08_egovernment_digital_services.txt, 08_egovernment_digital_services.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

The Welcome to Serbia portal is available in Serbian (Cyrillic and Latin), English, Russian, and Chinese.

### The passages the prompt carried

**1. 08_egovernment_digital_services.txt** (from character 1588)

```
The eUprava portal also connects to the eSanduče (electronic mailbox) service for receiving official documents and bills electronically.

WELCOME TO SERBIA PORTAL
Website: welcometoserbia.gov.rs

The official Government of Serbia portal specifically designed for foreign nationals. It provides:
- Comprehensive information about immigration to Serbia
- eApplication for C Visa (short-stay)
- eApplication for D Visa (long-stay)
- eApplication for Temporary Residence Permit (including Single Permit)
- Interactive guide: answer questions about your situation to get personalized instructions
- Help section with FAQs about residence, visas, and work permits

Languages available: Serbian (Cyrillic and Latin), English, Russian, Chinese

Key features:
- Online application submission for residence permits
- Document upload functionality
- Fee payment integration
- Application status tracking
- eDatabox (eSanduče) for receiving official decisions and notifications
```

**2. 07_visa_regime_entry.txt** (from character 5886)

```
USEFUL LINKS
- Ministry of Foreign Affairs — Visa Requirements: mfa.gov.rs/en/citizens/travel-serbia/visa-requirements
- Welcome to Serbia portal: welcometoserbia.gov.rs
- eVisa applications: evisa.welcometoserbia.gov.rs
- eApplication for temporary residence: eforeigner.welcometoserbia.gov.rs

Source: Ministry of Foreign Affairs (mfa.gov.rs), Welcome to Serbia portal (welcometoserbia.gov.rs), Ministry of Interior (mup.gov.rs)
```

**3. 08_egovernment_digital_services.txt** (from character 6116)

```
USEFUL CONTACTS
- Welcome to Serbia support: office@ite.gov.rs
- Tax Administration: purs.gov.rs/kontakt
- eUprava: euprava.gov.rs
- EPS customer service: eps.rs

Source: Government of the Republic of Serbia (euprava.gov.rs), Welcome to Serbia portal (welcometoserbia.gov.rs), Tax Administration (purs.gov.rs), EPS (eps.rs)
```

**4. 08_egovernment_digital_services.txt** (from character 1231)

```
eUPRAVA PORTAL (e-GOVERNMENT)
Website: euprava.gov.rs

The main e-government portal providing access to hundreds of administrative services:
- Address registration for foreigners (White Card)
- Scheduling appointments with government offices
- Generating payment orders for government fees
- Vehicle registration
- Tax-related services
- Document requests

The eUprava portal also connects to the eSanduče (electronic mailbox) service for receiving official documents and bills electronically.

WELCOME TO SERBIA PORTAL
Website: welcometoserbia.gov.rs
```

---

## 29. [ru] после обновления винды епорези перестал открываться. что делать?

- **Answer should come from:** 08_egovernment_digital_services.txt
- **Expected fact:** Использовать SmartBox 2.0.0 или вход через eID
- **Retrieved:** 08_egovernment_digital_services.txt, 08_egovernment_digital_services.txt, 08_egovernment_digital_services.txt, 08_egovernment_digital_services.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** yes

**Verdict:** [ ] grounded  [ ] ungrounded  [x] declined

**Notes:**

### The answer

I don't have enough information to answer that question.

### The passages the prompt carried

**1. 08_egovernment_digital_services.txt** (from character 427)

```
The eID portal is the central authentication system for accessing Serbian government e-services. To use most online government services, you must first create an account here.

How to register:
1. Go to eid.gov.rs
2. Choose registration method:
   - With a Serbian biometric ID card and card reader
   - With a qualified electronic certificate
   - With a foreign document (for foreigners with a residence permit)
3. Set up your credentials (username, password, two-factor authentication)
4. Once registered, you can access all connected government portals with a single login

For foreigners: Registration is possible once you have a foreigner registration number (EBS) and/or a biometric residence permit card. Some services may require the biometric card with a card reader for electronic signature.

eUPRAVA PORTAL (e-GOVERNMENT)
Website: euprava.gov.rs
```

**2. 08_egovernment_digital_services.txt** (from character 2555)

```
eFOREIGNER PORTAL
Website: eforeigner.welcometoserbia.gov.rs

Specifically for temporary residence and Single Permit applications:
- Submit residence applications electronically
- Submit changes of employer or employment basis
- Track application progress
- Receive decisions through the eDatabox
- Access help and FAQs

Important notes:
- Applications for minors (under 18) cannot be submitted online — must be done in person
- If the system asks for a Visa ID during extension and you do not have one, contact support at office@ite.gov.rs to have your account reset with your EBS number
- Once submitted, applications cannot be cancelled through the portal

ePOREZI PORTAL (eTAXES)
Website: eporezi.purs.gov.rs

The Tax Administration's electronic services portal:
- Submit tax returns electronically
- Monitor submitted returns
- View taxpayer account balances
- Grant/revoke authorization for tax representatives
- Available daily 6:00 AM to midnight
- Test environment available for practice
```

**3. 08_egovernment_digital_services.txt** (from character 5199)

```
Relevant services:
- Exchange rate information (middle rate used for tax calculations)
- Register of insurance companies
- Banking sector information
- Consumer protection in financial services

PRACTICAL TIPS FOR FOREIGNERS USING eGOVERNMENT
1. Start with eID registration as soon as you have your residence permit — it unlocks all other services.
2. Keep your foreigner registration number (EBS) readily available — you will need it for almost every online service.
3. Check your eSanduče regularly for official notifications and bills.
4. Many portals support English, but some forms and notifications may only be in Serbian — use browser translation tools if needed.
5. Save your payment reference numbers (poziv na broj) from bills — they are essential for correct payment attribution.
6. If you encounter technical issues, contact the respective portal's support via email before visiting an office in person.
```

**4. 08_egovernment_digital_services.txt** (from character 0)

```
eGOVERNMENT PORTALS AND DIGITAL SERVICES IN SERBIA

OVERVIEW
Serbia has developed several digital government platforms that allow both citizens and foreigners to access administrative services online. For immigrants, these portals are essential for managing residence permits, paying taxes, accessing utility services, and handling many bureaucratic tasks remotely.

eID PORTAL (ELECTRONIC IDENTIFICATION)
Website: eid.gov.rs

The eID portal is the central authentication system for accessing Serbian government e-services. To use most online government services, you must first create an account here.
```

---

## 30. [sr] hocu da otvorim firmu - sta se sve radi preko apr sajta?

- **Answer should come from:** 08_egovernment_digital_services.txt
- **Expected fact:** Registracija privrednog društva; registracija preduzetnika; registar stvarnih vlasnika; provera statusa
- **Retrieved:** 08_egovernment_digital_services.txt, 08_egovernment_digital_services.txt, 08_egovernment_digital_services.txt, 05_tax_registration_pib.txt
- **Document retrieved:** yes at position 1
- **Reads as a decline:** no

**Verdict:** [x] grounded  [ ] ungrounded  [ ] declined

**Notes:**

### The answer

Na APR (Agencija za privredne registre) sajtu možete obaviti sledeće usluge vezane za otvaranje firme u Srbiji:
- Registracija kompanije
- Registracija preduzetnika (sole proprietor)
- Pristup Registru stvarnih vlasnika
- Verifikacija statusa preduzeća

Sajt je delimično dostupan na engleskom jeziku.

### The passages the prompt carried

**1. 08_egovernment_digital_services.txt** (from character 1231)

```
eUPRAVA PORTAL (e-GOVERNMENT)
Website: euprava.gov.rs

The main e-government portal providing access to hundreds of administrative services:
- Address registration for foreigners (White Card)
- Scheduling appointments with government offices
- Generating payment orders for government fees
- Vehicle registration
- Tax-related services
- Document requests

The eUprava portal also connects to the eSanduče (electronic mailbox) service for receiving official documents and bills electronically.

WELCOME TO SERBIA PORTAL
Website: welcometoserbia.gov.rs
```

**2. 08_egovernment_digital_services.txt** (from character 1588)

```
The eUprava portal also connects to the eSanduče (electronic mailbox) service for receiving official documents and bills electronically.

WELCOME TO SERBIA PORTAL
Website: welcometoserbia.gov.rs

The official Government of Serbia portal specifically designed for foreign nationals. It provides:
- Comprehensive information about immigration to Serbia
- eApplication for C Visa (short-stay)
- eApplication for D Visa (long-stay)
- eApplication for Temporary Residence Permit (including Single Permit)
- Interactive guide: answer questions about your situation to get personalized instructions
- Help section with FAQs about residence, visas, and work permits

Languages available: Serbian (Cyrillic and Latin), English, Russian, Chinese

Key features:
- Online application submission for residence permits
- Document upload functionality
- Fee payment integration
- Application status tracking
- eDatabox (eSanduče) for receiving official decisions and notifications
```

**3. 08_egovernment_digital_services.txt** (from character 4529)

```
EPS ONLINE SERVICES
Website: eps.rs

Elektroprivreda Srbije online services:
- "Uvid u račun" (Bill overview): register to view and pay electricity bills
- Mobile app for iOS and Android
- View 3-year payment history
- Track consumption by zones and tariffs
- Pay via IPS Scan or card payments
- Opt for e-bills instead of paper (50 RSD discount)

APR (BUSINESS REGISTERS AGENCY)
Website: apr.gov.rs

For foreigners starting a business in Serbia:
- Company registration
- Sole proprietor (preduzetnik) registration
- Access to the Register of Beneficial Owners
- Business status verification
- Available partly in English

NATIONAL BANK OF SERBIA (NBS)
Website: nbs.rs

Relevant services:
- Exchange rate information (middle rate used for tax calculations)
- Register of insurance companies
- Banking sector information
- Consumer protection in financial services
```

**4. 05_tax_registration_pib.txt** (from character 4809)

```
USEFUL LINKS
- Tax Administration: purs.gov.rs
- eTaxes portal: eporezi.purs.gov.rs
- Freelancer portal: frilenseri.purs.gov.rs
- Freelancer guide (PDF): frilenseri.purs.gov.rs/upload/media/2024/3/28/11215/Guide_for_submitting_tax_application_and_payment_of_taxes_via_web_portal.pdf

Source: Tax Administration of the Republic of Serbia (purs.gov.rs), Welcome to Serbia portal (welcometoserbia.gov.rs)
```

---

## 31. [en] how do i swap my foreign driving licence for a serbian one?

- **Answer should come from:** nothing - not in the corpus
- **Expected fact:** Not in the corpus - the agent should decline
- **Retrieved:** 02_temporary_residence_permit.txt, 02_temporary_residence_permit.txt, 01_white_card_registration.txt, 03_opening_bank_account.txt
- **Document retrieved:** no
- **Reads as a decline:** yes

**Verdict:** [ ] grounded  [ ] ungrounded  [x] declined

**Notes:**

### The answer

I don't have enough information to answer that question.

### The passages the prompt carried

**1. 02_temporary_residence_permit.txt** (from character 1351)

```
HOW TO APPLY — STEP BY STEP
1. Register your address in Serbia (obtain the White Card / Beli karton).
2. Create an account on the Welcome to Serbia portal (welcometoserbia.gov.rs) or the eID portal (eid.gov.rs).
3. Submit the application electronically through the portal, or in person at the competent police department (MUP) responsible for foreigner affairs in your area.
4. Pay the required fees.
5. Wait for processing — the standard government processing time is 19 days for the Single Permit (significantly reduced from the previous 70 days).
6. If approved, visit the competent authority in person to provide biometric data (photo, fingerprints).
7. Receive your biometric residence permit card.
```

**2. 02_temporary_residence_permit.txt** (from character 2675)

```
Note: Proof of means of subsistence and health insurance are NOT required for foreigners who are immediate family members of a Serbian citizen or who apply on the basis of employment.

APPLYING ONLINE VS IN PERSON
Online (recommended):
- Register at welcometoserbia.gov.rs
- Fill out the online application form
- Upload required documents
- Pay fees electronically
- Track application status through the portal
- Receive notifications in your eDatabox (eSanduče)

In person:
- Schedule an appointment at the competent police department by email
- Submit the application with all required documents
- The address book of regional police departments can be found on the MUP website (mup.gov.rs)
```

**3. 01_white_card_registration.txt** (from character 0)

```
WHITE CARD (BELI KARTON) — REGISTRATION OF FOREIGNER'S ADDRESS IN SERBIA

WHAT IS THE WHITE CARD?
The White Card (Beli karton) is the mandatory registration of a foreigner's residential address in Serbia. Every foreigner staying in Serbia must register their address with the police within 24 hours of arrival. This is the first and most important step for any foreigner entering Serbia. Without it, you cannot proceed with any other administrative procedures such as obtaining a temporary residence permit, opening a bank account, or registering for health insurance.

WHO MUST REGISTER?
All foreign nationals staying in the Republic of Serbia, regardless of the purpose or duration of their stay. This applies to tourists, business visitors, and long-term residents alike.
```

**4. 03_opening_bank_account.txt** (from character 0)

```
OPENING A BANK ACCOUNT IN SERBIA AS A FOREIGNER

OVERVIEW
Foreign citizens can open bank accounts in Serbia, both for Serbian dinars (RSD) and foreign currencies (EUR, USD, CHF, GBP, and others). The process differs depending on whether you are classified as a resident or non-resident. As of January 2025, there are 20 licensed commercial banks operating in Serbia, including local, EU, Russian, American, Turkish, UAE, and Chinese banks.

RESIDENT VS NON-RESIDENT STATUS
Under the Law on Foreign Exchange Operations ("Official Gazette of RS" Nos. 62/2006, 31/2011, 119/2012, 139/2014, and 30/2018):
- Non-resident: A foreign citizen in the first year of their stay in Serbia.
- Resident: A foreign citizen who, on the basis of a residence permit or work visa, has stayed in Serbia for more than one year.

After one year, you acquire resident status and must notify the bank. The bank will close your non-resident account and open a resident account.

OPENING A NON-RESIDENT PERSONAL ACCOUNT
```

---

## 32. [ru] какие документы нужны чтобы отдать ребенка в сербскую школу?

- **Answer should come from:** nothing - not in the corpus
- **Expected fact:** Not in the corpus - the agent should decline
- **Retrieved:** 02_temporary_residence_permit.txt, 07_visa_regime_entry.txt, 02_temporary_residence_permit.txt, 07_visa_regime_entry.txt
- **Document retrieved:** no
- **Reads as a decline:** yes

**Verdict:** [ ] grounded  [ ] ungrounded  [x] declined

**Notes:**

### The answer

Я не имею достаточно информации, чтобы ответить на этот вопрос.

### The passages the prompt carried

**1. 02_temporary_residence_permit.txt** (from character 2056)

```
REQUIRED DOCUMENTS (GENERAL)
- Valid passport (validity must exceed the requested residence period by at least 3 months)
- Proof of registered residential address in Serbia (White Card)
- Landlord's statement consenting to address registration
- Proof of sufficient means of subsistence (employment contract, bank statement, or other evidence)
- Proof of health insurance valid in Serbia
- Additional documents depending on the specific ground for residence (employment contract, enrollment certificate, property deed, etc.)
- Passport-size color photo (less than 6 months old, light background, face clearly visible)

Note: Proof of means of subsistence and health insurance are NOT required for foreigners who are immediate family members of a Serbian citizen or who apply on the basis of employment.
```

**2. 07_visa_regime_entry.txt** (from character 5004)

```
Students:
Foreign students can obtain temporary residence on the basis of enrollment at a Serbian educational institution. Additional documents include enrollment certificates and proof of tuition payment.

Family Reunification:
Spouses and close family members of Serbian citizens or foreigners with approved residence can apply for temporary residence on the basis of family reunification.

IMPORTANT NOTES
- Always check the latest visa requirements for your specific nationality before traveling: mfa.gov.rs/en/citizens/travel-serbia/visa-requirements
- Overstaying your permitted period may result in fines, deportation, or entry bans.
- Your passport must be valid for at least 3 months beyond your intended stay.
- Keep your passport and White Card registration with you at all times.
- Border police may ask for proof of accommodation, return ticket, and sufficient funds.
```

**3. 02_temporary_residence_permit.txt** (from character 2675)

```
Note: Proof of means of subsistence and health insurance are NOT required for foreigners who are immediate family members of a Serbian citizen or who apply on the basis of employment.

APPLYING ONLINE VS IN PERSON
Online (recommended):
- Register at welcometoserbia.gov.rs
- Fill out the online application form
- Upload required documents
- Pay fees electronically
- Track application status through the portal
- Receive notifications in your eDatabox (eSanduče)

In person:
- Schedule an appointment at the competent police department by email
- Submit the application with all required documents
- The address book of regional police departments can be found on the MUP website (mup.gov.rs)
```

**4. 07_visa_regime_entry.txt** (from character 3166)

```
Required documents for visa application:
- Valid passport
- Passport-size photo (color, recent, light background)
- Proof of purpose of visit
- Proof of sufficient funds for accommodation and living expenses
- Proof of health insurance
- Evidence of intent to leave Serbia before visa expiry (return ticket, etc.)

For Visa D, additional documents matching the intended temporary residence basis are required (employment contract, enrollment certificate, property deed, etc.).

HOW TO APPLY FOR A VISA
1. Online: Through the Welcome to Serbia portal (evisa.welcometoserbia.gov.rs)
   - Register an account on the eID portal (eid.gov.rs)
   - Fill out the application form
   - Upload required documents
   - Pay the fee
   - Track status online

2. In person: At the Embassy or Consulate General of the Republic of Serbia in your country
   - Submit application with all required documents
   - Attend an interview if required
   - Processing time varies
```

---

## 33. [sr] koliko traje postupak za srpsko drzavljanstvo?

- **Answer should come from:** nothing - not in the corpus
- **Expected fact:** Not in the corpus - the agent should decline
- **Retrieved:** 02_temporary_residence_permit.txt, 02_temporary_residence_permit.txt, 02_temporary_residence_permit.txt, 07_visa_regime_entry.txt
- **Document retrieved:** no
- **Reads as a decline:** yes

**Verdict:** [ ] grounded  [ ] ungrounded  [x] declined

**Notes:**

### The answer

I don't have enough information to answer that question.

### The passages the prompt carried

**1. 02_temporary_residence_permit.txt** (from character 5055)

```
IMPORTANT NOTES
- You must stay in Serbia in accordance with the purpose for which residence was granted.
- If you entered with a Visa D, submit the residence application before the visa expires.
- Minor foreigners (under 18) must apply in person with a parent or guardian.
- All notifications regarding your application are delivered through the eDatabox if you applied online.
- The biometric residence permit card also serves as an identity document in Serbia.

AFTER 3 YEARS — PERMANENT RESIDENCE
After 3 continuous years of temporary residence, you may be eligible to apply for permanent residence (stalni boravak), which is granted indefinitely and issued as a foreigner ID card valid for 5 years (renewable).

LEGAL BASIS
Law on Foreigners ("Official Gazette of RS" No. 24/2018, 31/2019, and 62/2023)
Law on Employment of Foreigners
```

**2. 02_temporary_residence_permit.txt** (from character 1351)

```
HOW TO APPLY — STEP BY STEP
1. Register your address in Serbia (obtain the White Card / Beli karton).
2. Create an account on the Welcome to Serbia portal (welcometoserbia.gov.rs) or the eID portal (eid.gov.rs).
3. Submit the application electronically through the portal, or in person at the competent police department (MUP) responsible for foreigner affairs in your area.
4. Pay the required fees.
5. Wait for processing — the standard government processing time is 19 days for the Single Permit (significantly reduced from the previous 70 days).
6. If approved, visit the competent authority in person to provide biometric data (photo, fingerprints).
7. Receive your biometric residence permit card.
```

**3. 02_temporary_residence_permit.txt** (from character 3370)

```
FEES
Fees vary depending on the basis for residence. Examples:
- Standard temporary residence fee: approximately 20,660 RSD (as of 2024)
- Fee for residence based on marriage with a Serbian citizen: 11,350 RSD
- Biometric card issuance fee: additional charge applies
- Fees are paid via the eUprava portal or at a bank/post office

DURATION AND RENEWAL
- Temporary residence can be granted for up to 3 years and can be extended for the same period.
- The renewal application should be submitted at the earliest 60 days before the current permit expires.
- You must continue to meet the conditions for the approved basis of residence throughout its duration.
- If circumstances change (e.g., change of employer), you must notify the competent authority.
```

**4. 07_visa_regime_entry.txt** (from character 1718)

```
Warning: Serbian border authorities are aware of this practice. While it is technically legal if done within the 90/180 rule, repeatedly doing so without legitimate purpose may raise questions. If you plan to stay long-term, applying for a temporary residence permit is strongly recommended.

VISA TYPES

Type C — Short-Stay Visa:
- For stays up to 90 days
- For tourism, business visits, family visits, medical treatment, etc.
- Required for nationals of countries that do not have visa-free agreements with Serbia
- Applied for at Serbian embassies/consulates abroad or online via welcometoserbia.gov.rs
```

---

## 34. [en] what shots does my dog need to cross into serbia?

- **Answer should come from:** nothing - not in the corpus
- **Expected fact:** Not in the corpus - the agent should decline
- **Retrieved:** 07_visa_regime_entry.txt, 07_visa_regime_entry.txt, 04_health_insurance.txt, 04_health_insurance.txt
- **Document retrieved:** no
- **Reads as a decline:** yes

**Verdict:** [ ] grounded  [ ] ungrounded  [x] declined

**Notes:**

### The answer

I don't have enough information to answer that question.

### The passages the prompt carried

**1. 07_visa_regime_entry.txt** (from character 0)

```
VISA REGIME AND ENTRY REQUIREMENTS FOR SERBIA

OVERVIEW
Serbia has a relatively open visa policy. Many nationalities can enter visa-free for short stays. For longer stays, various visa types and residence permits are available. The entry requirements and visa regime are administered by the Ministry of Foreign Affairs and the Ministry of Interior.

GENERAL ENTRY REQUIREMENTS
When entering Serbia, all visitors must:
- Hold a valid passport (or national ID card for citizens of certain countries)
- Have sufficient funds for the duration of stay
- Have a return ticket or proof of onward travel (may be requested)
- Not be on the entry ban list
- Have no security concerns flagged by Serbian authorities

VISA-FREE ENTRY
Most foreigners, including citizens of the USA, UK, EU/EEA countries, Canada, Australia, and many others, can enter Serbia visa-free for up to 90 days within a 180-day period (counting from the first day of entry).
```

**2. 07_visa_regime_entry.txt** (from character 938)

```
Important: The 90/180 rule means you can stay a maximum of 90 days in any rolling 180-day period. This is not 90 consecutive days — it is cumulative. If you stay 90 days and leave, you cannot re-enter until 90 more days have passed.

VISA RUN
A "visa run" is the common practice of leaving Serbia before the 90-day limit and re-entering to reset the stay period. Common visa run destinations from Serbia include:
- Bosnia and Herzegovina (closest border crossings)
- North Macedonia
- Hungary
- Romania
- Bulgaria

Procedure: Drive or take transport to the border, exit Serbia (get an exit stamp), enter the neighboring country briefly, then re-enter Serbia (get a new entry stamp). Some organized visa runs involve walking across a border checkpoint and returning the same day.
```

**3. 04_health_insurance.txt** (from character 3004)

```
EMERGENCY MEDICAL CARE FOR FOREIGNERS
All foreign citizens in Serbia have the right to emergency medical assistance regardless of insurance status. However, payment depends on bilateral agreements:

Countries with bilateral health agreements (using European Health Insurance Card — EHIC):
- Austria, Bulgaria, Croatia, Hungary, Germany, Luxembourg, Slovakia, Slovenia, Czech Republic

Countries using specific bilateral forms:
- Belgium (Form BE/SRB 111), Netherlands (N/Y-111), Italy (IT-7), France (SE-21-03A/SE-21-04A), Montenegro (MNE/SRB-111), Bosnia and Herzegovina (BIH/SRB-111), North Macedonia (RM/SRB 111), Romania (Y/R 11), Turkey (TR/SRB 111)

Countries using insurance certificates:
- Poland and Great Britain — based on insurance certificates issued in their home country
```

**4. 04_health_insurance.txt** (from character 5179)

```
Exception: Proof of health insurance is NOT required for foreigners applying on the basis of employment (the employer handles insurance) or immediate family members of Serbian citizens.

EMERGENCY CONTACTS
- Emergency Ambulance Service: 194 (available 24 hours)
- In larger cities, emergency departments (urgentni centar) are open 24 hours
- Selected pharmacies (apoteke) operate 24 hours in major cities
- Note: Some medications require a doctor's prescription (recept)

RFZO CONTACT INFORMATION
- Website: rfzo.rs (Serbian), eng.rfzo.rs (English)
- Belgrade branch: contact through the website for specific branch addresses
- Niš office: Prijezdina 1, tel: +381 18 241-863
- Email contacts available on the RFZO website

LEGAL BASIS
Law on Health Insurance
Law on Health Care
Bilateral social security agreements (Serbia has agreements with 25+ countries)
```

---

## 35. [ru] где в белграде найти курсы сербского для иностранцев?

- **Answer should come from:** nothing - not in the corpus
- **Expected fact:** Not in the corpus - the agent should decline
- **Retrieved:** 07_visa_regime_entry.txt, 07_visa_regime_entry.txt, 08_egovernment_digital_services.txt, 03_opening_bank_account.txt
- **Document retrieved:** no
- **Reads as a decline:** yes

**Verdict:** [ ] grounded  [ ] ungrounded  [x] declined

**Notes:**

### The answer

I don't have enough information to answer that question.

### The passages the prompt carried

**1. 07_visa_regime_entry.txt** (from character 5004)

```
Students:
Foreign students can obtain temporary residence on the basis of enrollment at a Serbian educational institution. Additional documents include enrollment certificates and proof of tuition payment.

Family Reunification:
Spouses and close family members of Serbian citizens or foreigners with approved residence can apply for temporary residence on the basis of family reunification.

IMPORTANT NOTES
- Always check the latest visa requirements for your specific nationality before traveling: mfa.gov.rs/en/citizens/travel-serbia/visa-requirements
- Overstaying your permitted period may result in fines, deportation, or entry bans.
- Your passport must be valid for at least 3 months beyond your intended stay.
- Keep your passport and White Card registration with you at all times.
- Border police may ask for proof of accommodation, return ticket, and sufficient funds.
```

**2. 07_visa_regime_entry.txt** (from character 4122)

```
REGISTRATION UPON ARRIVAL
After entering Serbia, every foreigner must register their address within 24 hours:
- Hotels and accommodation facilities register automatically via the eTourist system
- Private hosts (landlords, friends, family) must register the foreigner at the nearest police station or via the eUprava portal
- This produces the White Card (Beli karton) — see the separate document on White Card registration

SPECIAL CATEGORIES

Digital Nomads:
Serbia has become a popular destination for digital nomads due to affordable living costs, good internet infrastructure, and a welcoming attitude toward remote workers. While Serbia does not yet have a specific "digital nomad visa," foreigners can:
- Stay visa-free for up to 90 days
- Apply for temporary residence on various grounds
- Register as freelancers for tax purposes through the frilenseri.purs.gov.rs portal
```

**3. 08_egovernment_digital_services.txt** (from character 1588)

```
The eUprava portal also connects to the eSanduče (electronic mailbox) service for receiving official documents and bills electronically.

WELCOME TO SERBIA PORTAL
Website: welcometoserbia.gov.rs

The official Government of Serbia portal specifically designed for foreign nationals. It provides:
- Comprehensive information about immigration to Serbia
- eApplication for C Visa (short-stay)
- eApplication for D Visa (long-stay)
- eApplication for Temporary Residence Permit (including Single Permit)
- Interactive guide: answer questions about your situation to get personalized instructions
- Help section with FAQs about residence, visas, and work permits

Languages available: Serbian (Cyrillic and Latin), English, Russian, Chinese

Key features:
- Online application submission for residence permits
- Document upload functionality
- Fee payment integration
- Application status tracking
- eDatabox (eSanduče) for receiving official decisions and notifications
```

**4. 03_opening_bank_account.txt** (from character 0)

```
OPENING A BANK ACCOUNT IN SERBIA AS A FOREIGNER

OVERVIEW
Foreign citizens can open bank accounts in Serbia, both for Serbian dinars (RSD) and foreign currencies (EUR, USD, CHF, GBP, and others). The process differs depending on whether you are classified as a resident or non-resident. As of January 2025, there are 20 licensed commercial banks operating in Serbia, including local, EU, Russian, American, Turkish, UAE, and Chinese banks.

RESIDENT VS NON-RESIDENT STATUS
Under the Law on Foreign Exchange Operations ("Official Gazette of RS" Nos. 62/2006, 31/2011, 119/2012, 139/2014, and 30/2018):
- Non-resident: A foreign citizen in the first year of their stay in Serbia.
- Resident: A foreign citizen who, on the basis of a residence permit or work visa, has stayed in Serbia for more than one year.

After one year, you acquire resident status and must notify the bank. The bank will close your non-resident account and open a resident account.

OPENING A NON-RESIDENT PERSONAL ACCOUNT
```

---

## 36. [en] can i get married here as a foreigner? what do i need

- **Answer should come from:** nothing - not in the corpus
- **Expected fact:** Not in the corpus - the agent should decline
- **Retrieved:** 02_temporary_residence_permit.txt, 01_white_card_registration.txt, 02_temporary_residence_permit.txt, 02_temporary_residence_permit.txt
- **Document retrieved:** no
- **Reads as a decline:** yes

**Verdict:** [ ] grounded  [ ] ungrounded  [x] declined

**Notes:**

### The answer

I don't have enough information to answer that question.

### The passages the prompt carried

**1. 02_temporary_residence_permit.txt** (from character 4124)

```
SINGLE PERMIT (UNIFIED PERMIT FOR RESIDENCE AND WORK)
Since February 2024, foreigners employed in Serbia receive a Single Permit that combines both residence and work authorization. Key points:
- Applied for through the Welcome to Serbia portal
- Standard processing time: 19 days
- Visa-free nationals must apply for a Single Permit (Visa D is no longer available for most visa-free nationals who want to work)
- The employer can submit the application on behalf of the employee with a power of attorney
- Changes of employer or basis of employment can be submitted through the portal

CERTIFICATE OF INITIATED PROCEDURE
After submitting your application, you receive a certificate confirming that the procedure has been initiated. This certificate includes your foreigner registration number (EBS), which you need for exercising rights and obligations in Serbia (opening a bank account, registering for health insurance, etc.).
```

**2. 01_white_card_registration.txt** (from character 1333)

```
HOW TO REGISTER — STEP BY STEP
1. The landlord (property owner) or accommodation provider must go to the nearest police station (policijska stanica) in the territory where the accommodation is located.
2. Required documents:
   - Foreigner's passport (original for inspection, photocopy to submit)
   - Landlord's personal ID card (lična karta)
   - Proof of property ownership or rental agreement
   - Landlord's written statement (izjava stanodavca) consenting to register the foreigner at their address
3. The police station issues the White Card (Beli karton) — a registration slip confirming the foreigner's registered address.
4. Keep this document safe — you will need it for virtually every subsequent administrative procedure.
```

**3. 02_temporary_residence_permit.txt** (from character 2675)

```
Note: Proof of means of subsistence and health insurance are NOT required for foreigners who are immediate family members of a Serbian citizen or who apply on the basis of employment.

APPLYING ONLINE VS IN PERSON
Online (recommended):
- Register at welcometoserbia.gov.rs
- Fill out the online application form
- Upload required documents
- Pay fees electronically
- Track application status through the portal
- Receive notifications in your eDatabox (eSanduče)

In person:
- Schedule an appointment at the competent police department by email
- Submit the application with all required documents
- The address book of regional police departments can be found on the MUP website (mup.gov.rs)
```

**4. 02_temporary_residence_permit.txt** (from character 5772)

```
LEGAL BASIS
Law on Foreigners ("Official Gazette of RS" No. 24/2018, 31/2019, and 62/2023)
Law on Employment of Foreigners

USEFUL LINKS AND CONTACTS
- Welcome to Serbia portal: welcometoserbia.gov.rs
- eApplication for temporary residence: eforeigner.welcometoserbia.gov.rs
- Ministry of Interior (MUP): mup.gov.rs
- Support email: office@ite.gov.rs

Source: Welcome to Serbia portal (welcometoserbia.gov.rs), Ministry of Interior of the Republic of Serbia (mup.gov.rs)
```

---

## 37. [sr] kolika je kazna za prekoracenje brzine?

- **Answer should come from:** nothing - not in the corpus
- **Expected fact:** Not in the corpus - the agent should decline
- **Retrieved:** 07_visa_regime_entry.txt, 02_temporary_residence_permit.txt, 07_visa_regime_entry.txt, 06_utility_bills_payments.txt
- **Document retrieved:** no
- **Reads as a decline:** yes

**Verdict:** [ ] grounded  [ ] ungrounded  [x] declined

**Notes:**

### The answer

I don't have enough information to answer that question.

### The passages the prompt carried

**1. 07_visa_regime_entry.txt** (from character 938)

```
Important: The 90/180 rule means you can stay a maximum of 90 days in any rolling 180-day period. This is not 90 consecutive days — it is cumulative. If you stay 90 days and leave, you cannot re-enter until 90 more days have passed.

VISA RUN
A "visa run" is the common practice of leaving Serbia before the 90-day limit and re-entering to reset the stay period. Common visa run destinations from Serbia include:
- Bosnia and Herzegovina (closest border crossings)
- North Macedonia
- Hungary
- Romania
- Bulgaria

Procedure: Drive or take transport to the border, exit Serbia (get an exit stamp), enter the neighboring country briefly, then re-enter Serbia (get a new entry stamp). Some organized visa runs involve walking across a border checkpoint and returning the same day.
```

**2. 02_temporary_residence_permit.txt** (from character 3370)

```
FEES
Fees vary depending on the basis for residence. Examples:
- Standard temporary residence fee: approximately 20,660 RSD (as of 2024)
- Fee for residence based on marriage with a Serbian citizen: 11,350 RSD
- Biometric card issuance fee: additional charge applies
- Fees are paid via the eUprava portal or at a bank/post office

DURATION AND RENEWAL
- Temporary residence can be granted for up to 3 years and can be extended for the same period.
- The renewal application should be submitted at the earliest 60 days before the current permit expires.
- You must continue to meet the conditions for the approved basis of residence throughout its duration.
- If circumstances change (e.g., change of employer), you must notify the competent authority.
```

**3. 07_visa_regime_entry.txt** (from character 1718)

```
Warning: Serbian border authorities are aware of this practice. While it is technically legal if done within the 90/180 rule, repeatedly doing so without legitimate purpose may raise questions. If you plan to stay long-term, applying for a temporary residence permit is strongly recommended.

VISA TYPES

Type C — Short-Stay Visa:
- For stays up to 90 days
- For tourism, business visits, family visits, medical treatment, etc.
- Required for nationals of countries that do not have visa-free agreements with Serbia
- Applied for at Serbian embassies/consulates abroad or online via welcometoserbia.gov.rs
```

**4. 06_utility_bills_payments.txt** (from character 2192)

```
Discount: Pay before the 15th of the month to receive approximately 5-10% discount on the next bill.

INFORMATIKA (NOVI SAD) — COMMUNAL SERVICES
In Novi Sad, JKP "Informatika" performs a similar unified billing function.
- Office: 3 Car Lazar Blvd., 21102 Novi Sad
- Contact Centre: 0800 222 021
- Late payments incur interest calculated per the National Bank of Serbia rate.
- Enforced collection may be initiated for unpaid bills after a warning letter.

INTERNET AND PHONE
Major providers:
- MTS (Telekom Srbija): largest provider, offers fiber optic, ADSL, mobile, and TV packages
- Yettel (formerly Telenor): mobile and internet services
- A1 (formerly Vip Mobile): mobile and home internet
- SBB (Serbia Broadband): cable internet and TV

SIM card: You can purchase a SIM card without documents, but activation and full use require a valid passport. Visit the operator's office to link the card to your passport.
```

---

## 38. [ru] какая сейчас минималка в сербии?

- **Answer should come from:** nothing - not in the corpus
- **Expected fact:** Not in the corpus - the agent should decline
- **Retrieved:** 03_opening_bank_account.txt, 03_opening_bank_account.txt, 03_opening_bank_account.txt, 06_utility_bills_payments.txt
- **Document retrieved:** no
- **Reads as a decline:** yes

**Verdict:** [ ] grounded  [ ] ungrounded  [x] declined

**Notes:**

### The answer

I don't have enough information to answer that question.

### The passages the prompt carried

**1. 03_opening_bank_account.txt** (from character 3879)

```
IMPORTANT NOTES
- Banks in Serbia are regulated by the National Bank of Serbia (NBS), established in 1884.
- Deposits in Serbian banks are insured (deposit guarantee scheme).
- Banks must comply with anti-money laundering regulations but cannot refuse service on discriminatory grounds (citizenship, gender, religion). If refused, the bank must provide reasons in writing.
- Some banks have limited reasons for opening non-resident accounts (e.g., specific business purposes).
- A Tax Identification Number (PIB) may be required by certain banks — especially for freelancers or business owners.
- Having a local bank account is essential for: receiving salary, paying utility bills, paying taxes, and daily transactions.
- The IPS (Instant Payment System) enables instant transfers between Serbian bank accounts.
```

**2. 03_opening_bank_account.txt** (from character 0)

```
OPENING A BANK ACCOUNT IN SERBIA AS A FOREIGNER

OVERVIEW
Foreign citizens can open bank accounts in Serbia, both for Serbian dinars (RSD) and foreign currencies (EUR, USD, CHF, GBP, and others). The process differs depending on whether you are classified as a resident or non-resident. As of January 2025, there are 20 licensed commercial banks operating in Serbia, including local, EU, Russian, American, Turkish, UAE, and Chinese banks.

RESIDENT VS NON-RESIDENT STATUS
Under the Law on Foreign Exchange Operations ("Official Gazette of RS" Nos. 62/2006, 31/2011, 119/2012, 139/2014, and 30/2018):
- Non-resident: A foreign citizen in the first year of their stay in Serbia.
- Resident: A foreign citizen who, on the basis of a residence permit or work visa, has stayed in Serbia for more than one year.

After one year, you acquire resident status and must notify the bank. The bank will close your non-resident account and open a resident account.

OPENING A NON-RESIDENT PERSONAL ACCOUNT
```

**3. 03_opening_bank_account.txt** (from character 2976)

```
Other banks: Erste Bank, UniCredit (now OTP), Mobi Banka (fully mobile banking), AIK Banka, NLB Banka, Poštanska Štedionica

FEES AND COSTS
- Account opening: Usually free of charge (some premium banks like Raiffeisen may charge up to EUR 1,000 for corporate accounts)
- Monthly maintenance: Typically 5-10 EUR for RSD accounts; foreign currency account maintenance is often free
- International transfers: Fees vary by bank — compare before choosing
- Card issuance: Usually included in the account package
- Serbia is becoming part of SEPA from 2025, which will simplify EUR transfers

BANKING SERVICES AVAILABLE
- Online banking (most banks offer English language support)
- Mobile banking applications
- Debit cards (Visa, MasterCard, DinaCard accepted locally)
- International SWIFT transfers
- Standing orders for automatic bill payments
- Multi-currency accounts (RSD, EUR, USD, CHF, GBP, etc.)
```

**4. 06_utility_bills_payments.txt** (from character 0)

```
UTILITY BILLS AND PAYMENTS IN SERBIA — GUIDE FOR FOREIGNERS

OVERVIEW
Living in Serbia means dealing with several regular utility bills. The main ones are electricity (EPS), communal services including heating and water (Infostan in Belgrade, Informatika in Novi Sad), internet/phone, and building maintenance. This guide explains what each bill covers, how to read them, and how to pay.

ELECTRICITY — EPS (ELEKTROPRIVREDA SRBIJE)
EPS is the sole electricity provider in Serbia. Every household has an EPS account.

What the bill includes:
- Electricity consumption measured in kWh
- Charges may vary by tariff zone (day/night rates for dual-tariff meters)
- Monthly fixed connection fee

Typical costs:
- Summer months (no electric heating): 3,000-3,500 RSD (25-30 EUR)
- Winter months (with electric heating): 15,000-18,000 RSD (130-155 EUR)
- Serbia has among the cheapest electricity prices in Europe
```

---

## 39. [en] thinking of buying a flat here as a foreigner - what tax do i pay on the purchase?

- **Answer should come from:** nothing - not in the corpus
- **Expected fact:** Not in the corpus - the agent should decline
- **Retrieved:** 05_tax_registration_pib.txt, 05_tax_registration_pib.txt, 05_tax_registration_pib.txt, 05_tax_registration_pib.txt
- **Document retrieved:** no
- **Reads as a decline:** yes

**Verdict:** [ ] grounded  [ ] ungrounded  [x] declined

**Notes:**

### The answer

I don't have enough information to answer that question.

### The passages the prompt carried

**1. 05_tax_registration_pib.txt** (from character 1693)

```
ePOREZI (eTAXES) PORTAL
The ePorezi portal provides electronic tax services:
- Submit electronically signed tax return forms
- Monitor the status of submitted returns
- View the balance of taxpayer accounts
- Grant or revoke authorization for electronic services
- Available daily from 6:00 AM to midnight
- A TEST ENVIRONMENT is available for learning and verification

INCOME TAXES FOR FOREIGNERS

Employed foreigners:
- If you are employed by a Serbian company, your employer calculates and pays income tax and social contributions on your behalf through the Individual Tax Return (PPP-PD form).
- Tax rates: 10% income tax on salary, plus mandatory social contributions (pension: 25%, health: 10.3%)

Freelancers (self-taxation):
Since December 2022, Serbia has regulated freelancer taxation through amendments to the Law on Personal Income Tax.
```

**2. 05_tax_registration_pib.txt** (from character 2862)

```
Important: You do NOT need a work permit to register as a freelancer and report income under the self-taxation regime. You only need a foreigner's registration number (EBS) or a PIB.

How to report freelancer income:
1. Register on the Freelancers portal: frilenseri.purs.gov.rs
2. Submit quarterly tax returns (PP OPO-K form):
   - Q1: April 1-30
   - Q2: July 1-30
   - Q3: October 1-30
   - Q4: January 1-30 (of the following year)
3. The portal automatically generates a payment slip with QR code for easy payment via mobile banking.
4. Convert foreign currency income to RSD using the National Bank of Serbia middle exchange rate on the payment date.

Freelancer tax rates:
- Income tax: 20% on the net taxable amount (after deducting normative costs)
- Normative costs (automatic deduction): depends on income type (typically 20-50%)
- Pension and disability insurance: 24% (mandatory)
- Health insurance: 10.3% (mandatory, unless covered by a bilateral agreement)
```

**3. 05_tax_registration_pib.txt** (from character 2398)

```
Freelancers (self-taxation):
Since December 2022, Serbia has regulated freelancer taxation through amendments to the Law on Personal Income Tax.

Who qualifies as a freelancer for tax purposes:
- Serbian residents earning income from foreign payers (companies, entrepreneurs, or individuals abroad)
- Foreign residents (non-residents) earning income in Serbia from foreign payers
- Anyone earning income from a domestic natural person who does not withhold taxes

Important: You do NOT need a work permit to register as a freelancer and report income under the self-taxation regime. You only need a foreigner's registration number (EBS) or a PIB.
```

**4. 05_tax_registration_pib.txt** (from character 3834)

```
If you paid tax abroad on the same income, you can claim a tax credit (up to the amount of Serbian tax liability) to avoid double taxation. Serbia has double taxation avoidance agreements with over 65 countries.

VISA RUN AND TAX IMPLICATIONS
Foreigners who stay in Serbia on visa-free entry (up to 90 days in 180 days) and perform "visa runs" (exiting and re-entering the country to reset the stay period) should be aware that earning income while in Serbia may create tax obligations regardless of visa status.

IMPORTANT NOTES
- The PIB is used as the VAT number for businesses engaged in taxable transactions.
- Verify your PIB through the Tax Administration's official verification tool.
- If you are a sole proprietor, you must register for personal salary payments within 5 days of receiving your PIB.
- All tax obligations can be managed electronically through the ePorezi portal.
- Late tax payments incur interest calculated per the National Bank of Serbia rates.
```

---

## 40. [sr] kako da prijavim kradju policiji i dobijem zapisnik?

- **Answer should come from:** nothing - not in the corpus
- **Expected fact:** Not in the corpus - the agent should decline
- **Retrieved:** 01_white_card_registration.txt, 05_tax_registration_pib.txt, 04_health_insurance.txt, 06_utility_bills_payments.txt
- **Document retrieved:** no
- **Reads as a decline:** yes

**Verdict:** [ ] grounded  [ ] ungrounded  [x] declined

**Notes:**

### The answer

I don't have enough information to answer that question.

### The passages the prompt carried

**1. 01_white_card_registration.txt** (from character 1333)

```
HOW TO REGISTER — STEP BY STEP
1. The landlord (property owner) or accommodation provider must go to the nearest police station (policijska stanica) in the territory where the accommodation is located.
2. Required documents:
   - Foreigner's passport (original for inspection, photocopy to submit)
   - Landlord's personal ID card (lična karta)
   - Proof of property ownership or rental agreement
   - Landlord's written statement (izjava stanodavca) consenting to register the foreigner at their address
3. The police station issues the White Card (Beli karton) — a registration slip confirming the foreigner's registered address.
4. Keep this document safe — you will need it for virtually every subsequent administrative procedure.
```

**2. 05_tax_registration_pib.txt** (from character 759)

```
HOW TO OBTAIN A PIB

For individuals (including foreigners):
1. Online: Through the ePorezi (eTaxes) portal at purs.gov.rs
2. In person: At a local Tax Administration office
3. Required documents:
   - Valid passport or ID card
   - Proof of address in Serbia (residence permit or White Card)
   - Foreigner registration number (EBS) if applicable
4. There is no fee for PIB registration.

For businesses:
- The PIB is automatically assigned when registering a company with the Serbian Business Registers Agency (APR — Agencija za privredne registre).
- The PIB also serves as the VAT number for businesses registered for VAT.

TAX ADMINISTRATION (PORESKA UPRAVA)
- The Tax Administration is a body within the Ministry of Finance.
- It carries out assessment, control, and collection of public revenues.
- Head Office: Cara Dušana 145, Belgrade-Zemun
- Website: purs.gov.rs (Serbian and English)
- eTaxes portal: eporezi.purs.gov.rs
```

**3. 04_health_insurance.txt** (from character 1646)

```
2. Required documents:
   - Copy of passport (main page and page with residence permit sticker) or printout of biometric residence card (boravak)
   - Paid receipt for health card production (approximately 400 RSD per person, including children)
   - Completed application form (Form MA)
   - For sole proprietors: extract from the business register (Izvod o registraciji) and sometimes a tax office certificate
   - For employees: employer's confirmation of employment registration

3. Visit the appropriate RFZO branch and submit documents.
4. You will receive a temporary paper document that allows you to seek medical help immediately.
5. After processing, you receive a health insurance card (zdravstvena kartica).
```

**4. 06_utility_bills_payments.txt** (from character 2937)

```
SIM card: You can purchase a SIM card without documents, but activation and full use require a valid passport. Visit the operator's office to link the card to your passport.

BUILDING MAINTENANCE (ODRŽAVANJE ZGRADE)
- A separate fee for common areas, elevator maintenance, and building cleaning
- Charged per square meter of your apartment
- Usually collected through the Infostan/Informatika bill or separately by the building management (skupština stanara)

HOW TO PAY UTILITY BILLS

Method 1: Post Office (Pošta)
- Bring the paper bill (nalog za uplatu) to any post office
- Pay in cash (RSD only)
- Lowest commission fees (starting from 65 RSD per payment)
- Available at all post offices across Serbia

Method 2: Exchange Offices (Menjačnica)
- Many exchange offices (such as Exclusive Change) accept bill payments
- Commission: typically 0.60%, minimum 65 RSD
- Available evenings and weekends at many locations
```

---
